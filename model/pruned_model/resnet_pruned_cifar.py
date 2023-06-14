"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_preserved_filter_num(mask):
    return int(mask.sum())


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResBasicBlock_pruned(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, masks, stride=1):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.masks = masks
        preserved_filter_num1 = get_preserved_filter_num(masks[0])
        self.conv1 = conv3x3(inplanes, preserved_filter_num1, stride)
        self.bn1 = nn.BatchNorm2d(preserved_filter_num1)
        self.relu = nn.ReLU(inplace=True)
        preserved_filter_num2 = get_preserved_filter_num(masks[1])
        self.conv2 = conv3x3(
            preserved_filter_num1,
            preserved_filter_num2,
        )
        self.bn2 = nn.BatchNorm2d(preserved_filter_num2)
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

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # padding 0 for feature map to get the same shape of short cut
        shortcut_out = self.shortcut(x).clone()
        padded_out = torch.zeros_like(shortcut_out).clone()
        # with torch.no_grad():
        #     for padded_feature_map, feature_map in zip(padded_out, out):
        #         padded_feature_map[self.masks[1] == 1] = feature_map
        for padded_feature_map, feature_map in zip(padded_out, out):
            padded_feature_map[self.masks[1] == 1] = feature_map

        assert padded_out.shape == shortcut_out.shape, "wrong shape"

        padded_out += shortcut_out
        padded_out = self.relu(padded_out)

        return padded_out


class ResNet_pruned(nn.Module):
    def __init__(self, block, num_layers, masks=[], num_classes=10):
        super().__init__()
        assert (num_layers - 2) % 6 == 0, "depth should be 6n+2"
        assert len(masks) == num_layers - 2, "masks should be (num_layers - 2)"
        n = (num_layers - 2) // 6

        self.inplanes = 16
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            block, 16, blocks=n, stride=1, masks=masks[0 : 2 * n]
        )
        self.layer2 = self._make_layer(
            block, 32, blocks=n, stride=2, masks=masks[2 * n : 4 * n]
        )
        self.layer3 = self._make_layer(
            block, 64, blocks=n, stride=2, masks=masks[4 * n : 6 * n]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, masks):
        layers = []

        layers.append(block(self.inplanes, planes, masks[0:2], stride))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, masks[2 * i : 2 * i + 2]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        for block in self.layer1:
            x = block(x)

        for block in self.layer2:
            x = block(x)

        for block in self.layer3:
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_56_pruned_cifar10(masks):
    return ResNet_pruned(
        block=ResBasicBlock_pruned, num_layers=56, masks=masks, num_classes=10
    )


def resnet_110_pruned_cifar10(masks):
    return ResNet_pruned(
        block=ResBasicBlock_pruned, num_layers=110, masks=masks, num_classes=10
    )


def resnet_56_pruned_cifar100(masks):
    return ResNet_pruned(
        block=ResBasicBlock_pruned, num_layers=56, masks=masks, num_classes=100
    )
