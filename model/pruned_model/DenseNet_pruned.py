import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_preserved_filter_num(mask):
    return int(mask.sum())


class SingleLayer_pruned(nn.Module):
    def __init__(self, nChannels, growthRate, mask):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        preserved_filter_num = get_preserved_filter_num(mask)
        self.conv1 = nn.Conv2d(
            nChannels, preserved_filter_num, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition_pruned(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet_pruned(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck, masks=[]):
        super().__init__()

        assert len(masks) == depth - 2, "masks should be (depth - 2)"

        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(
            nChannels, growthRate, nDenseBlocks, bottleneck, masks[0:nDenseBlocks]
        )

        nChannels += sum(
            [get_preserved_filter_num(masks[i]) for i in range(0, nDenseBlocks)]
        )

        nOutChannels = get_preserved_filter_num(masks[nDenseBlocks])
        self.trans1 = Transition_pruned(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(
            nChannels,
            growthRate,
            nDenseBlocks,
            bottleneck,
            masks[nDenseBlocks + 1 : 2 * nDenseBlocks + 1],
        )

        nChannels += sum(
            [
                get_preserved_filter_num(masks[i])
                for i in range(nDenseBlocks + 1, 2 * nDenseBlocks + 1)
            ]
        )

        nOutChannels = get_preserved_filter_num(masks[2 * nDenseBlocks + 1])
        self.trans2 = Transition_pruned(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(
            nChannels,
            growthRate,
            nDenseBlocks,
            bottleneck,
            masks[2 * nDenseBlocks + 2 : 3 * nDenseBlocks + 2],
        )

        nChannels += sum(
            [
                get_preserved_filter_num(masks[i])
                for i in range(2 * nDenseBlocks + 2, 3 * nDenseBlocks + 2)
            ]
        )

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, masks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(SingleLayer_pruned(nChannels, growthRate, masks[i]))
            nChannels += get_preserved_filter_num(masks[i])
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []

        out = self.conv1(x)
        out = self.dense1(out)
        feature_list.append(out)
        out = self.trans1(out)
        out = self.dense2(out)
        feature_list.append(out)
        out = self.trans2(out)
        out = self.dense3(out)
        feature_list.append(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = self.fc(out)
        return out, feature_list


def DenseNet_40_pruned_cifar10(masks):
    return DenseNet_pruned(
        growthRate=12,
        depth=40,
        reduction=1,
        nClasses=10,
        bottleneck=False,
        masks=masks,
    )
