"""GoogLeNet with PyTorch."""
import torch
import torch.nn as nn

import copy
import math
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
        # self.epoch += 1
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
            Flops_total + 32 * 32 * 3 * 3 * 3 * 192 + 32 * 32 * 192
        )  # first conv and bn layer
        block_input_num = 192
        for i, m in enumerate(self.mask_modules):
            if i % 7 == 0 or i % 7 == 1 or i % 7 == 3 or i % 7 == 6:
                Flops_conv = (
                    m.feature_map_h
                    * m.feature_map_w
                    * m.kernel_size
                    * m.kernel_size
                    * block_input_num
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
            if i % 7 == 6:
                block_input_num = (
                    self.mask_modules[i].mask.sum()
                    + self.mask_modules[i - 1].mask.sum()
                    + self.mask_modules[i - 4].mask.sum()
                    + self.mask_modules[i - 6].mask.sum()
                )
            Flops_bn = m.feature_map_h * m.feature_map_w * m.mask.sum()
            Flops_total = Flops_total + Flops_conv + Flops_bn
        return Flops_total


class Inception_sparse(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super().__init__()
        self.n1x1 = n1x1
        self.n3x3 = n3x3
        self.n5x5 = n5x5
        self.pool_planes = pool_planes

        # 1x1 conv branch
        if self.n1x1:
            self.branch1x1 = nn.Sequential(
                SoftMaskedConv2d(in_planes, n1x1, kernel_size=1, bias=True),
                nn.BatchNorm2d(n1x1),
                nn.ReLU(True),
            )

        # 1x1 conv -> 3x3 conv branch
        if self.n3x3:
            self.branch3x3 = nn.Sequential(
                SoftMaskedConv2d(in_planes, n3x3red, kernel_size=1, bias=True),
                nn.BatchNorm2d(n3x3red),
                nn.ReLU(True),
                SoftMaskedConv2d(n3x3red, n3x3, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(n3x3),
                nn.ReLU(True),
            )

        # 1x1 conv -> 5x5 conv branch
        if self.n5x5 > 0:
            self.branch5x5 = nn.Sequential(
                SoftMaskedConv2d(in_planes, n5x5red, kernel_size=1, bias=True),
                nn.BatchNorm2d(n5x5red),
                nn.ReLU(True),
                SoftMaskedConv2d(n5x5red, n5x5, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(n5x5),
                nn.ReLU(True),
                SoftMaskedConv2d(n5x5, n5x5, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(n5x5),
                nn.ReLU(True),
            )

        # 3x3 pool -> 1x1 conv branch
        if self.pool_planes > 0:
            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                SoftMaskedConv2d(in_planes, pool_planes, kernel_size=1, bias=True),
                nn.BatchNorm2d(pool_planes),
                nn.ReLU(True),
            )

    def forward(self, x, ticket):
        out = []
        if self.n1x1:
            # y1 = self.branch1x1(x)
            y1 = x.clone()
            for layer in self.branch1x1:
                if isinstance(layer, SoftMaskedConv2d):
                    y1 = layer(y1, ticket)
                else:
                    y1 = layer(y1)
            out.append(y1)
        if self.n3x3:
            # y2 = self.branch3x3(x)
            y2 = x.clone()
            for layer in self.branch3x3:
                if isinstance(layer, SoftMaskedConv2d):
                    y2 = layer(y2, ticket)
                else:
                    y2 = layer(y2)
            out.append(y2)
        if self.n5x5:
            # y3 = self.branch5x5(x)
            y3 = x.clone()
            for layer in self.branch5x5:
                if isinstance(layer, SoftMaskedConv2d):
                    y3 = layer(y3, ticket)
                else:
                    y3 = layer(y3)
            out.append(y3)
        if self.pool_planes:
            # y4 = self.branch_pool(x)
            y4 = x.clone()
            for layer in self.branch_pool:
                if isinstance(layer, SoftMaskedConv2d):
                    y4 = layer(y4, ticket)
                else:
                    y4 = layer(y4)
            out.append(y4)
        return torch.cat(out, 1)


class GoogLeNet_sparse(MaskedNet):
    def __init__(
        self,
        block=Inception_sparse,
        filters=None,
        num_classes=10,
        gumbel_start_temperature=2,
        gumbel_end_temperature=0.1,
        num_epochs=350,
    ):
        super().__init__(gumbel_start_temperature, gumbel_end_temperature, num_epochs)
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

        self.covert1 = nn.Conv2d(
            filters[1][0] + filters[1][1] + filters[1][2] + filters[1][3],
            filters[1][0] + filters[1][1] + filters[1][2] + filters[1][3],
            kernel_size=1,
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

        self.covert2 = nn.Conv2d(
            filters[6][0] + filters[6][1] + filters[6][2] + filters[6][3],
            filters[6][0] + filters[6][1] + filters[6][2] + filters[6][3],
            kernel_size=1,
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

        self.covert3 = nn.Conv2d(
            filters[8][0] + filters[8][1] + filters[8][2] + filters[8][3],
            filters[8][0] + filters[8][1] + filters[8][2] + filters[8][3],
            kernel_size=1,
        )

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(sum(filters[-1]), num_classes)

        self.mask_modules = [m for m in self.modules() if type(m) == SoftMaskedConv2d]

    def forward(self, x):
        feature_list = []

        out = self.pre_layers(x)
        # 192 x 32 x 32
        out = self.inception_a3(out, self.ticket)
        # 256 x 32 x 32
        out = self.inception_b3(out, self.ticket)

        feature_list.append(self.covert1(out))

        # 480 x 32 x 32
        out = self.maxpool1(out)
        # 480 x 16 x 16
        out = self.inception_a4(out, self.ticket)
        # 512 x 16 x 16
        out = self.inception_b4(out, self.ticket)
        # 512 x 16 x 16
        out = self.inception_c4(out, self.ticket)
        # 512 x 16 x 16
        out = self.inception_d4(out, self.ticket)
        # 528 x 16 x 16
        out = self.inception_e4(out, self.ticket)

        feature_list.append(self.covert2(out))

        # 823 x 16 x 16
        out = self.maxpool2(out)
        # 823 x 8 x 8
        out = self.inception_a5(out, self.ticket)
        # 823 x 8 x 8
        out = self.inception_b5(out, self.ticket)

        feature_list.append(self.covert3(out))

        # 1024 x 8 x 8
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, feature_list


def GoogLeNet_sparse_cifar10(
    gumbel_start_temperature=2, gumbel_end_temperature=0.5, num_epochs=350
):
    return GoogLeNet_sparse(
        block=Inception_sparse,
        num_classes=10,
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
    )
