from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .streaming import StreamingContainer
from .streaming_conv import StreamingAvgPool1d, StreamingConv1d, StreamingConvTranspose1d
from .utils import init_weights


class StreamingDownSamplingBlock(StreamingContainer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_weight_norm: bool = True,
        use_shortcut: bool = True,
    ):
        super().__init__()
        self.use_shortcut = bool(use_shortcut)
        self.down_conv = StreamingConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            use_weight_norm=use_weight_norm,
        )
        if self.use_shortcut:
            self.pool = StreamingAvgPool1d(in_channels, stride)
            self.shortcut_conv = (
                StreamingConv1d(in_channels, out_channels, kernel_size=1, use_weight_norm=use_weight_norm)
                if in_channels != out_channels
                else None
            )
        else:
            self.pool = None
            self.shortcut_conv = None
        self.down_conv.apply(init_weights)
        if self.shortcut_conv is not None:
            self.shortcut_conv.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_shortcut:
            return self.down_conv(x)
        assert self.pool is not None
        shortcut = self.pool(x)
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
        return self.down_conv(x) + shortcut


class StreamingUpSamplingBlock(StreamingContainer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_weight_norm: bool = True,
        use_shortcut: bool = True,
    ):
        super().__init__()
        self.stride = int(stride)
        self.use_shortcut = bool(use_shortcut)
        self.up_conv = StreamingConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            use_weight_norm=use_weight_norm,
        )
        if self.use_shortcut:
            self.shortcut_conv = (
                StreamingConv1d(in_channels, out_channels, kernel_size=1, use_weight_norm=use_weight_norm)
                if in_channels != out_channels
                else None
            )
        else:
            self.shortcut_conv = None
        self.up_conv.apply(init_weights)
        if self.shortcut_conv is not None:
            self.shortcut_conv.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_shortcut:
            return self.up_conv(x)
        shortcut = x.repeat_interleave(self.stride, dim=-1)
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
        return self.up_conv(x) + shortcut


class StreamingConvNeXtBlock(StreamingContainer):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        if adanorm_num_embeddings is not None:
            raise NotImplementedError("Adaptive LayerNorm is not used by this codec streaming path.")
        self.dwconv = StreamingConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x.transpose(1, 2))
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x
