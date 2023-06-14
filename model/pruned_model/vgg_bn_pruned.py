import torch
import torch.nn as nn
from collections import OrderedDict
import math

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


def get_preserved_filter_num(mask):
    return int(mask.sum())


class VGG_pruned(nn.Module):
    def __init__(self, num_classes=10, init_weights=True, cfg=None, masks=[]):
        super().__init__()
        self.masks = masks
        self.features = nn.Sequential()

        if cfg is None:
            cfg = defaultcfg

        self.cfg = cfg
        self.features = self.make_layers(cfg[:-1], masks)
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

        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, masks=[]):
        layers = nn.Sequential()
        preserved_filter_num_before = 3
        num = 0
        for i, v in enumerate(cfg):
            if v == "M":
                layers.add_module("pool%d" % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                preserved_filter_num = get_preserved_filter_num(masks[num])
                conv2d = nn.Conv2d(
                    preserved_filter_num_before,
                    preserved_filter_num,
                    kernel_size=3,
                    padding=1,
                )

                layers.add_module("conv%d" % i, conv2d)
                layers.add_module("norm%d" % i, nn.BatchNorm2d(preserved_filter_num))
                layers.add_module("relu%d" % i, nn.ReLU(inplace=True))
                preserved_filter_num_before = preserved_filter_num
                num += 1

        return layers

    def forward(self, x):
        x = self.features(x)

        padded_out = torch.zeros(
            [x.shape[0], self.cfg[-2], x.shape[2], x.shape[3]]
        ).clone()
        for padded_feature_map, feature_map in zip(padded_out, x):
            padded_feature_map[self.masks[-1] == 1] = feature_map

        x = padded_out
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
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def VGG_16_bn_pruned_cifar10(masks):
    model = VGG_pruned(num_classes=10, init_weights=True, cfg=defaultcfg, masks=masks)
    return model
