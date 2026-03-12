import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["BilinearConvTranspose2d", "NearestConvTranspose2d"]


class BilinearConvTranspose2d(nn.Module):
    """A conv transpose initialized to bilinear interpolation.
    this conv tranpose isn't equal to biilinear interpolation mathematically.

    other open source reference: https://gist.github.com/Sundrops/4d4f73f58166b984f5c6bb1723d4e627
    """

    def __init__(self, channels, scale_factor=2, with_groups=True, freeze_weights=True):
        super().__init__()

        assert isinstance(
            scale_factor, int
        ), f"{type(self).__name__} only supports interger scale factor, while gets {scale_factor}"

        if with_groups:
            groups = channels
        else:
            groups = 1

        ksize = 2 * scale_factor - scale_factor % 2
        pad = math.ceil((scale_factor - 1) / 2.0)
        self.upsample = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=ksize,
            stride=scale_factor,
            padding=pad,
            groups=groups,
            bias=False,
        )

        self.init_weights(freeze_weights)

    def init_weights(self, freeze_weights):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                out_channels, in_channels, kh, kw = m.weight.size()
                m.weight.data.copy_(
                    self.get_upsampling_weight(in_channels, out_channels, kh)
                )
                if freeze_weights:
                    m.weight.requires_grad = False

    @staticmethod
    def get_upsampling_weight(in_channels, out_channels, kernel_size):
        assert (in_channels == 1) or (in_channels == out_channels)
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros(
            (out_channels, in_channels, kernel_size, kernel_size), dtype=np.float64
        )
        weight[list(range(out_channels)), list(range(in_channels)), :, :] = filt
        return torch.from_numpy(weight).float()

    def forward(self, x):
        return self.upsample(x)

class NearestConvTranspose2d(nn.Module):
    """
    A ConvTranspose2d for the implementation of the nearest interpolation.
    """
    def __init__(self, channels, scale_factor=2, with_groups=False, freeze_weights=True):
        super().__init__()

        assert isinstance(
            scale_factor, int
        ), f"{type(self).__name__} only supports interger scale factor, while gets {scale_factor}"
        assert scale_factor >= 2, \
            f"{type(self).__name__} only supports interger scale factor at least 2, while gets {scale_factor}"

        if with_groups:
            groups = channels
        else:
            groups = 1

        kernel_size = scale_factor
        stride = scale_factor
        padding = 0  # (kernel_size - stride) / 2

        self.upsample = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.init_weights(freeze_weights, with_groups)

    def init_weights(self, freeze_weights, with_groups):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                out_channels, in_channels, kh, kw = m.weight.size()
                m.weight.data.copy_(
                    self.get_upsampling_weight(out_channels, in_channels, kh, with_groups)
                )
                if freeze_weights:
                    m.weight.requires_grad = False

    @staticmethod
    def get_upsampling_weight(out_channels, in_channels, kernel_size, with_groups):
        if with_groups: # depthwise
            assert in_channels == 1
            weight = torch.ones(out_channels, 1, kernel_size, kernel_size)
            # weight[:, :, 1:kernel_size-1, 1:kernel_size-1] = 1
        else:
            assert in_channels == out_channels
            weight = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
            for _c_idx in range(out_channels):
                # weight[_c_idx, _c_idx, 1:kernel_size-1, 1:kernel_size-1] = 1
                weight[_c_idx, _c_idx, :, :] = 1
        return weight

    def forward(self, x):
        return self.upsample(x)
