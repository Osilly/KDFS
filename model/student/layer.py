import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-x)))


def hard_sigmod(x):
    return torch.min(torch.max(x + 0.5, torch.zeros_like(x)), torch.ones_like(x))


class SoftMaskedConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = None

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.init_weight()

        self.init_mask()

        self.mask = torch.ones([self.out_channels, 1, 1, 1])
        self.gumbel_temperature = 1

        self.feature_map_h = 0
        self.feature_map_w = 0

    def init_weight(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(self.out_channels, 2, 1, 1))
        nn.init.kaiming_normal_(self.mask_weight)

    def compute_mask(self, ticket):
        if ticket:
            mask = torch.argmax(self.mask_weight, dim=1).unsqueeze(1).float()
        else:
            mask = F.gumbel_softmax(
                logits=self.mask_weight, tau=self.gumbel_temperature, hard=True, dim=1
            )[:, 1, :, :].unsqueeze(1)

        return mask  # shape = [C, 1, 1, 1]

    def update_gumbel_temperature(self, gumbel_temperature):
        self.gumbel_temperature = gumbel_temperature

    def forward(self, x, ticket=False):
        self.mask = self.compute_mask(ticket)
        masked_weight = self.weight * self.mask
        out = F.conv2d(
            x,
            weight=masked_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        self.feature_map_h, self.feature_map_w = out.shape[2], out.shape[3]
        return out

    def extra_repr(self):
        return "{}, {}, kernel_size={}, stride={}, padding={}".format(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
        )
