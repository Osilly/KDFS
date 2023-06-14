"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch
import torch.nn as nn
import math

__all__ = ["mobilenetv2"]


def get_preserved_filter_num(mask):
    return int(mask.sum())


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn_pruned(inp, oup, mask):
    return nn.Sequential(
        nn.Conv2d(inp, get_preserved_filter_num(mask), 1, 1, 0, bias=False),
        nn.BatchNorm2d(get_preserved_filter_num(mask)),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual_pruned(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, masks):
        super().__init__()
        assert stride in [1, 2]

        self.expand_ratio = expand_ratio
        self.masks = masks

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(
                    hidden_dim,
                    get_preserved_filter_num(masks[0]),
                    1,
                    1,
                    0,
                    bias=False,
                ),
                nn.BatchNorm2d(get_preserved_filter_num(masks[0])),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, get_preserved_filter_num(masks[0]), 1, 1, 0, bias=False),
                nn.BatchNorm2d(get_preserved_filter_num(masks[0])),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    get_preserved_filter_num(masks[0]),
                    get_preserved_filter_num(masks[0]),
                    3,
                    stride,
                    1,
                    groups=get_preserved_filter_num(masks[0]),
                    bias=False,
                ),
                nn.BatchNorm2d(get_preserved_filter_num(masks[0])),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(
                    get_preserved_filter_num(masks[0]),
                    get_preserved_filter_num(masks[1]),
                    1,
                    1,
                    0,
                    bias=False,
                ),
                nn.BatchNorm2d(get_preserved_filter_num(masks[1])),
            )

    def forward(self, x):
        out = self.conv(x)

        if self.identity:
            # padding 0 for feature map to get the same shape of short cut
            shortcut_out = x.clone()
            padded_out = torch.zeros_like(shortcut_out).clone()
            if self.expand_ratio == 1:
                for padded_feature_map, feature_map in zip(padded_out, out):
                    padded_feature_map[self.masks[0] == 1] = feature_map
            else:
                for padded_feature_map, feature_map in zip(padded_out, out):
                    padded_feature_map[self.masks[1] == 1] = feature_map

            assert padded_out.shape == shortcut_out.shape, "wrong shape"

            return x + padded_out
        else:
            if self.expand_ratio == 1:
                padded_out = torch.zeros(
                    [out.shape[0], len(self.masks[0]), out.shape[2], out.shape[3]]
                ).clone()
                for padded_feature_map, feature_map in zip(padded_out, out):
                    padded_feature_map[self.masks[0] == 1] = feature_map
            else:
                padded_out = torch.zeros(
                    [out.shape[0], len(self.masks[1]), out.shape[2], out.shape[3]]
                ).clone()
                for padded_feature_map, feature_map in zip(padded_out, out):
                    padded_feature_map[self.masks[1] == 1] = feature_map

            return padded_out


class MobileNetV2_pruned(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, masks=[]):
        super().__init__()
        assert len(masks) == 34, "len of masks should be 34"

        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual_pruned

        num = 0
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(
                c * width_mult, 4 if width_mult == 0.1 else 8
            )
            for i in range(n):
                if num == 0:
                    current_masks = masks[0:1]
                else:
                    current_masks = masks[1 + (num - 1) * 2 : 1 + (num - 1) * 2 + 2]
                num += 1

                layers.append(
                    block(
                        input_channel,
                        output_channel,
                        s if i == 0 else 1,
                        t,
                        masks=current_masks,
                    )
                )
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = (
            _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8)
            if width_mult > 1.0
            else 1280
        )
        self.conv = conv_1x1_bn_pruned(input_channel, output_channel, mask=masks[33])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(get_preserved_filter_num(masks[33]), num_classes)

        self._initialize_weights()

    def forward(self, x):
        feature_list = []

        for i, block in enumerate(self.features):
            x = block(x)
            if i == 1:
                feature_list.append(x)
            elif i == 6:
                feature_list.append(x)
            elif i == 13:
                feature_list.append(x)
            elif i == 17:
                feature_list.append(x)
        # x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, feature_list

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def MobileNetV2_pruned_imagenet(masks):
    return MobileNetV2_pruned(num_classes=1000, width_mult=1.0, masks=masks)
