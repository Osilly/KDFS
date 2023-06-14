"""GoogLeNet with PyTorch."""
import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super().__init__()
        self.n1x1 = n1x1
        self.n3x3 = n3x3
        self.n5x5 = n5x5
        self.pool_planes = pool_planes

        # 1x1 conv branch
        if self.n1x1:
            self.branch1x1 = nn.Sequential(
                nn.Conv2d(in_planes, n1x1, kernel_size=1),
                nn.BatchNorm2d(n1x1),
                nn.ReLU(True),
            )

        # 1x1 conv -> 3x3 conv branch
        if self.n3x3:
            self.branch3x3 = nn.Sequential(
                nn.Conv2d(in_planes, n3x3red, kernel_size=1),
                nn.BatchNorm2d(n3x3red),
                nn.ReLU(True),
                nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
                nn.BatchNorm2d(n3x3),
                nn.ReLU(True),
            )

        # 1x1 conv -> 5x5 conv branch
        if self.n5x5 > 0:
            self.branch5x5 = nn.Sequential(
                nn.Conv2d(in_planes, n5x5red, kernel_size=1),
                nn.BatchNorm2d(n5x5red),
                nn.ReLU(True),
                nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
                nn.BatchNorm2d(n5x5),
                nn.ReLU(True),
                nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
                nn.BatchNorm2d(n5x5),
                nn.ReLU(True),
            )

        # 3x3 pool -> 1x1 conv branch
        if self.pool_planes > 0:
            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_planes, pool_planes, kernel_size=1),
                nn.BatchNorm2d(pool_planes),
                nn.ReLU(True),
            )

    def forward(self, x):
        out = []
        if self.n1x1:
            y1 = self.branch1x1(x)
            out.append(y1)
        if self.n3x3:
            y2 = self.branch3x3(x)
            out.append(y2)
        if self.n5x5:
            y3 = self.branch5x5(x)
            out.append(y3)
        if self.pool_planes:
            y4 = self.branch_pool(x)
            out.append(y4)
        return torch.cat(out, 1)


class GoogLeNet(nn.Module):
    def __init__(self, block=Inception, filters=None, num_classes=10):
        super().__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        if filters is None:
            filters = [
                [64, 128, 32, 32],
                [128, 192, 96, 64],
                [192, 208, 48, 64],
                [160, 224, 64, 64],
                [128, 256, 64, 64],
                [112, 288, 64, 64],
                [256, 320, 128, 128],
                [256, 320, 128, 128],
                [384, 384, 128, 128],
            ]

        self.inception_a3 = block(
            192,
            filters[0][0],
            96,
            filters[0][1],
            16,
            filters[0][2],
            filters[0][3],
        )
        self.inception_b3 = block(
            sum(filters[0]),
            filters[1][0],
            128,
            filters[1][1],
            32,
            filters[1][2],
            filters[1][3],
        )

        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception_a4 = block(
            sum(filters[1]),
            filters[2][0],
            96,
            filters[2][1],
            16,
            filters[2][2],
            filters[2][3],
        )
        self.inception_b4 = block(
            sum(filters[2]),
            filters[3][0],
            112,
            filters[3][1],
            24,
            filters[3][2],
            filters[3][3],
        )
        self.inception_c4 = block(
            sum(filters[3]),
            filters[4][0],
            128,
            filters[4][1],
            24,
            filters[4][2],
            filters[4][3],
        )
        self.inception_d4 = block(
            sum(filters[4]),
            filters[5][0],
            144,
            filters[5][1],
            32,
            filters[5][2],
            filters[5][3],
        )
        self.inception_e4 = block(
            sum(filters[5]),
            filters[6][0],
            160,
            filters[6][1],
            32,
            filters[6][2],
            filters[6][3],
        )

        self.inception_a5 = block(
            sum(filters[6]),
            filters[7][0],
            160,
            filters[7][1],
            32,
            filters[7][2],
            filters[7][3],
        )
        self.inception_b5 = block(
            sum(filters[7]),
            filters[8][0],
            192,
            filters[8][1],
            48,
            filters[8][2],
            filters[8][3],
        )

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(sum(filters[-1]), num_classes)

    def forward(self, x):
        feature_list = []

        out = self.pre_layers(x)
        # 192 x 32 x 32
        out = self.inception_a3(out)
        # 256 x 32 x 32
        out = self.inception_b3(out)

        feature_list.append(out)

        # 480 x 32 x 32
        out = self.maxpool1(out)
        # 480 x 16 x 16
        out = self.inception_a4(out)
        # 512 x 16 x 16
        out = self.inception_b4(out)
        # 512 x 16 x 16
        out = self.inception_c4(out)
        # 512 x 16 x 16
        out = self.inception_d4(out)
        # 528 x 16 x 16
        out = self.inception_e4(out)

        feature_list.append(out)

        # 823 x 16 x 16
        out = self.maxpool2(out)
        # 823 x 8 x 8
        out = self.inception_a5(out)
        # 823 x 8 x 8
        out = self.inception_b5(out)

        feature_list.append(out)

        # 1024 x 8 x 8
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, feature_list


def GoogLeNet_cifar10():
    return GoogLeNet(block=Inception, num_classes=10)
