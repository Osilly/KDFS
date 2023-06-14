import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from .layer import SoftMaskedConv2d


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
        nDenseBlocks = (self.depth - 4) // 3
        nChannels = 2 * self.growthRate

        Flops_total = (
            Flops_total + 32 * 32 * 3 * 3 * 3 * nChannels + 32 * 32 * nChannels
        )  # first conv and bn layer
        for i, m in enumerate(self.mask_modules):
            if i == 0:
                Flops_conv = (
                    m.feature_map_h
                    * m.feature_map_w
                    * m.kernel_size
                    * m.kernel_size
                    * m.in_channels
                    * m.mask.sum()
                )
                nChannels = nChannels + m.mask.sum()
            elif i in [nDenseBlocks, 2 * nDenseBlocks + 1]:
                Flops_conv = (
                    m.feature_map_h
                    * m.feature_map_w
                    * m.kernel_size
                    * m.kernel_size
                    * nChannels
                    * m.mask.sum()
                )
                nChannels = m.mask.sum()
            else:
                Flops_conv = (
                    m.feature_map_h
                    * m.feature_map_w
                    * m.kernel_size
                    * m.kernel_size
                    * nChannels
                    * m.mask.sum()
                )
                nChannels = nChannels + m.mask.sum()
            Flops_bn = m.feature_map_h * m.feature_map_w * m.mask.sum()
            Flops_total = Flops_total + Flops_conv + Flops_bn
        return Flops_total


class SingleLayer_sparse(nn.Module):
    def __init__(self, nChannels, growthRate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = SoftMaskedConv2d(
            nChannels, growthRate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x, ticket):
        out = self.conv1(F.relu(self.bn1(x)), ticket)
        out = torch.cat((x, out), 1)
        return out


class Transition_sparse(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = SoftMaskedConv2d(
            nChannels, nOutChannels, kernel_size=1, bias=False
        )

    def forward(self, x, ticket):
        out = self.conv1(F.relu(self.bn1(x)), ticket)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet_sparse(MaskedNet):
    def __init__(
        self,
        growthRate,
        depth,
        reduction,
        nClasses,
        bottleneck,
        gumbel_start_temperature=2,
        gumbel_end_temperature=0.1,
        num_epochs=350,
    ):
        super().__init__(gumbel_start_temperature, gumbel_end_temperature, num_epochs)

        self.depth = depth
        self.growthRate = growthRate
        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate

        # convert feature
        self.covert1 = nn.Conv2d(nChannels, nChannels, kernel_size=1)

        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition_sparse(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate

        # convert feature
        self.covert2 = nn.Conv2d(nChannels, nChannels, kernel_size=1)

        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition_sparse(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate

        # convert feature
        self.covert3 = nn.Conv2d(nChannels, nChannels, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        self.mask_modules = [m for m in self.modules() if type(m) == SoftMaskedConv2d]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, SoftMaskedConv2d):
                n = m.kernel_size * m.kernel_size * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(SingleLayer_sparse(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []

        out = self.conv1(x)
        for block in self.dense1:
            out = block(out, self.ticket)
        feature_list.append(self.covert1(out))
        out = self.trans1(out, self.ticket)
        for block in self.dense2:
            out = block(out, self.ticket)
        feature_list.append(self.covert2(out))
        out = self.trans2(out, self.ticket)
        for block in self.dense3:
            out = block(out, self.ticket)
        feature_list.append(self.covert3(out))
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = self.fc(out)
        return out, feature_list


def DenseNet_40_sparse_cifar10(
    gumbel_start_temperature=2, gumbel_end_temperature=0.5, num_epochs=350
):
    return DenseNet_sparse(
        growthRate=12,
        depth=40,
        reduction=1,
        nClasses=10,
        bottleneck=False,
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
    )
