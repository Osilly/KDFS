import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import SoftMaskedConv2d
import copy


def conv3x3(in_planes, out_planes, stride=1):
    return SoftMaskedConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class MaskedNet(nn.Module):
    def __init__(
        self, gumbel_start_temperature=2, gumbel_end_temperature=0.1, num_epochs=350
    ):
        super().__init__()
        self.gumbel_start_temperature = gumbel_start_temperature
        self.gumbel_end_temperature = gumbel_end_temperature
        self.num_epochs = num_epochs

        self.gumbel_temperature = gumbel_start_temperature
        
        self.ticket = False

        self.mask_modules = []

    def checkpoint(self):
        for m in self.mask_modules:
            m.checkpoint()
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.Linear)
            ):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules:
            m.rewind_weights()
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.Linear)
            ):
                m.load_state_dict(m.checkpoint)


    def update_gumbel_temperature(self, epoch):
        self.gumbel_temperature = self.gumbel_start_temperature * math.pow(
            self.gumbel_end_temperature / self.gumbel_start_temperature,
            epoch / self.num_epochs,
        )
        for m in self.mask_modules:
            m.update_gumbel_temperature(self.gumbel_temperature)

    def get_flops(self):
        Flops_total = torch.tensor(0)
        # Flops_conv = feature_map_h * feature_map_w * k * k * c_in * c_out
        # Flops_bn = feature_map_h * feature_map_w * c_in
        Flops_total = (
            Flops_total + 32 * 32 * 3 * 3 * 3 * 16 + 32 * 32 * 16
        )  # first conv and bn layer
        for i, m in enumerate(self.mask_modules):
            if i % 2 == 0:
                Flops_conv = (
                    m.feature_map_h
                    * m.feature_map_w
                    * m.kernel_size
                    * m.kernel_size
                    * m.in_channels
                    * m.mask.sum()
                )
            else:
                Flops_conv = (
                    m.feature_map_h
                    * m.feature_map_w
                    * m.kernel_size
                    * m.kernel_size
                    * self.mask_modules[i - 1].mask.sum()
                    * m.mask.sum()
                )
            Flops_bn = m.feature_map_h * m.feature_map_w * m.mask.sum()
            Flops_total = Flops_total + Flops_conv + Flops_bn
        return Flops_total


class ResBasicBlock_sparse(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4),
                    "constant",
                    0,
                )
            )

    def forward(self, x, ticket):
        out = self.conv1(x, ticket)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, ticket)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet_sparse(MaskedNet):
    def __init__(
        self,
        block,
        num_layers,
        num_classes=10,
        gumbel_start_temperature=2,
        gumbel_end_temperature=0.1,
        num_epochs=350,
    ):
        super().__init__(gumbel_start_temperature, gumbel_end_temperature, num_epochs)
        assert (num_layers - 2) % 6 == 0, "depth should be 6n+2"
        n = (num_layers - 2) // 6

        self.inplanes = 16
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, blocks=n, stride=1)
        self.layer2 = self._make_layer(block, 32, blocks=n, stride=2)
        self.layer3 = self._make_layer(block, 64, blocks=n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # convert feature
        self.covert1 = nn.Conv2d(16, 16, kernel_size=1)
        self.covert2 = nn.Conv2d(32, 32, kernel_size=1)
        self.covert3 = nn.Conv2d(64, 64, kernel_size=1)

        self.mask_modules = [m for m in self.modules() if type(m) == SoftMaskedConv2d]

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, SoftMaskedConv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        layers = []

        layers.append(block(self.inplanes, planes, stride))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        for block in self.layer1:
            x = block(x, self.ticket)
        feature_list.append(self.covert1(x))

        for block in self.layer2:
            x = block(x, self.ticket)
        feature_list.append(self.covert2(x))

        for block in self.layer3:
            x = block(x, self.ticket)
        feature_list.append(self.covert3(x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, feature_list


def resnet_56_sparse_cifar10(
    gumbel_start_temperature=2, gumbel_end_temperature=0.5, num_epochs=350
):
    return ResNet_sparse(
        block=ResBasicBlock_sparse,
        num_layers=56,
        num_classes=10,
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
    )


def resnet_110_sparse_cifar10(
    gumbel_start_temperature=2, gumbel_end_temperature=0.5, num_epochs=350
):
    return ResNet_sparse(
        block=ResBasicBlock_sparse,
        num_layers=110,
        num_classes=10,
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
    )


def resnet_56_sparse_cifar100(
    gumbel_start_temperature=2, gumbel_end_temperature=0.5, num_epochs=350
):
    return ResNet_sparse(
        block=ResBasicBlock_sparse,
        num_layers=56,
        num_classes=100,
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
    )
