import copy
import torch
import torch.nn as nn
from collections import OrderedDict
import math

from .layer import SoftMaskedConv2d

norm_mean, norm_var = 0.0, 1.0

defaultcfg = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    512,
]


class MaskedNet(nn.Module):
    def __init__(
        self, gumbel_start_temperature=2, gumbel_end_temperature=0.1, num_epochs=350
    ):
        super().__init__()
        self.gumbel_start_temperature = gumbel_start_temperature
        self.gumbel_end_temperature = gumbel_end_temperature
        self.num_epochs = num_epochs

        self.gumbel_temperature = gumbel_start_temperature

        # self.epoch = 0
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

    # def prune(self):
    #     for m in self.mask_modules:
    #         m.prune()

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
        for i, m in enumerate(self.mask_modules):
            if i == 0:
                Flops_conv = (
                    m.feature_map_h
                    * m.feature_map_w
                    * m.kernel_size
                    * m.kernel_size
                    * 3
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


class VGG_sparse(MaskedNet):
    def __init__(
        self,
        num_classes=10,
        init_weights=True,
        cfg=None,
        gumbel_start_temperature=2,
        gumbel_end_temperature=0.1,
        num_epochs=350,
    ):
        super().__init__(gumbel_start_temperature, gumbel_end_temperature, num_epochs)
        self.features = nn.Sequential()

        if cfg is None:
            cfg = defaultcfg

        self.features = self.make_layers(cfg[:-1])
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(cfg[-2], cfg[-1])),
                    ("norm1", nn.BatchNorm1d(cfg[-1])),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(cfg[-1], num_classes)),
                ]
            )
        )

        self.mask_modules = [m for m in self.modules() if type(m) == SoftMaskedConv2d]

        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg):
        layers = nn.Sequential()
        in_channels = 3
        for i, v in enumerate(cfg):
            if v == "M":
                layers.add_module("pool%d" % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = SoftMaskedConv2d(
                    in_channels, v, kernel_size=3, padding=1, bias=True
                )

                layers.add_module("conv%d" % i, conv2d)
                layers.add_module("norm%d" % i, nn.BatchNorm2d(v))
                layers.add_module("relu%d" % i, nn.ReLU(inplace=True))
                in_channels = v

        return layers

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, [torch.zeros_like(x)]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, SoftMaskedConv2d):
                n = m.kernel_size * m.kernel_size * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def VGG_16_bn_sparse_cifar10(
    gumbel_start_temperature=2,
    gumbel_end_temperature=0.1,
    num_epochs=350,
):
    model = VGG_sparse(
        num_classes=10,
        init_weights=True,
        cfg=defaultcfg,
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
    )
    return model
