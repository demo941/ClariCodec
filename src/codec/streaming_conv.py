from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

from .streaming import StreamingModule, StreamingState


def _maybe_weight_norm(module: nn.Module, enabled: bool) -> nn.Module:
    return weight_norm(module) if enabled else module


@dataclass
class StreamingConv1dState(StreamingState):
    previous: torch.Tensor

    def reset(self) -> None:
        self.previous.zero_()


class StreamingConv1d(StreamingModule[StreamingConv1dState]):
    """Strict-causal Conv1d with Mimi-style history cache.

    Non-streaming forward is also causal and equivalent to one streaming pass
    over the same sequence, as long as chunk sizes are multiples of stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn(
                "StreamingConv1d with stride > 1 and dilation > 1 can be hard to reason about "
                f"(kernel_size={kernel_size}, stride={stride}, dilation={dilation})."
            )
        self.conv = _maybe_weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            use_weight_norm,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight  # type: ignore[attr-defined]

    @property
    def bias(self) -> torch.Tensor | None:
        return self.conv.bias  # type: ignore[attr-defined]

    @property
    def stride(self) -> int:
        return self.conv.stride[0]  # type: ignore[attr-defined]

    @property
    def kernel_size(self) -> int:
        return self.conv.kernel_size[0]  # type: ignore[attr-defined]

    @property
    def dilation(self) -> int:
        return self.conv.dilation[0]  # type: ignore[attr-defined]

    @property
    def effective_kernel_size(self) -> int:
        return (self.kernel_size - 1) * self.dilation + 1

    @property
    def history_size(self) -> int:
        return max(0, self.effective_kernel_size - self.stride)

    def _init_streaming_state(self, batch_size: int) -> StreamingConv1dState:
        param = next(self.parameters())
        previous = torch.zeros(
            batch_size,
            self.conv.in_channels,  # type: ignore[attr-defined]
            self.history_size,
            dtype=param.dtype,
            device=param.device,
        )
        return StreamingConv1dState(batch_size, param.device, previous)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")
        if x.shape[-1] == 0:
            return x.new_empty(x.shape[0], self.conv.out_channels, 0)  # type: ignore[attr-defined]
        if x.shape[-1] % self.stride != 0:
            raise ValueError(
                f"StreamingConv1d input length must be a multiple of stride={self.stride}, "
                f"got T={x.shape[-1]}"
            )

        state = self._streaming_state
        if state is None:
            history = x.new_zeros(x.shape[0], x.shape[1], self.history_size)
        else:
            history = state.previous

        if self.history_size:
            x_in = torch.cat([history, x], dim=-1)
        else:
            x_in = x

        y = self.conv(x_in)

        if state is not None and self.history_size:
            state.previous.copy_(x_in[..., -self.history_size :])
        return y


@dataclass
class StreamingLookaheadConv1dState(StreamingState):
    history: torch.Tensor
    pending: torch.Tensor

    def reset(self) -> None:
        self.history.zero_()
        self.pending = self.pending[..., :0]


class StreamingLookaheadConv1d(StreamingModule[StreamingLookaheadConv1dState]):
    """Fixed-delay lookahead Conv1d for codec-frame latent features.

    The non-streaming path pads both sides and preserves input length. The
    streaming path delays output by ``lookahead`` frames, caching both the left
    context and the not-yet-safe pending frames.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        lookahead: int,
        bias: bool = True,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()
        self.lookahead = int(lookahead)
        self.kernel_size = int(kernel_size)
        if self.lookahead <= 0:
            raise ValueError(f"lookahead must be > 0, got {lookahead}")
        if self.lookahead >= self.kernel_size:
            raise ValueError(f"lookahead must be < kernel_size, got {lookahead} >= {kernel_size}")
        self.left_context = self.kernel_size - 1 - self.lookahead
        self.conv = _maybe_weight_norm(
            nn.Conv1d(
                channels,
                channels,
                self.kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=bias,
            ),
            use_weight_norm,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight  # type: ignore[attr-defined]

    @property
    def bias(self) -> torch.Tensor | None:
        return self.conv.bias  # type: ignore[attr-defined]

    def _init_streaming_state(self, batch_size: int) -> StreamingLookaheadConv1dState:
        param = next(self.parameters())
        history = torch.zeros(
            batch_size,
            self.conv.in_channels,  # type: ignore[attr-defined]
            self.left_context,
            dtype=param.dtype,
            device=param.device,
        )
        pending = torch.zeros(
            batch_size,
            self.conv.in_channels,  # type: ignore[attr-defined]
            0,
            dtype=param.dtype,
            device=param.device,
        )
        return StreamingLookaheadConv1dState(batch_size, param.device, history, pending)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")
        if x.shape[-1] == 0:
            return x.new_empty(x.shape[0], self.conv.out_channels, 0)  # type: ignore[attr-defined]

        state = self._streaming_state
        if state is None:
            x = F.pad(x, (self.left_context, self.lookahead))
            return self.conv(x)
        return self._forward_streaming(x, state)

    def _forward_streaming(self, x: torch.Tensor, state: StreamingLookaheadConv1dState) -> torch.Tensor:
        state.pending = torch.cat([state.pending, x], dim=-1) if state.pending.shape[-1] else x
        safe_len = state.pending.shape[-1] - self.lookahead
        if safe_len <= 0:
            return x.new_empty(x.shape[0], self.conv.out_channels, 0)  # type: ignore[attr-defined]

        x_in = torch.cat([state.history, state.pending], dim=-1) if self.left_context else state.pending
        y = self.conv(x_in)
        if y.shape[-1] != safe_len:
            raise RuntimeError(f"Internal length mismatch: expected {safe_len}, got {y.shape[-1]}")

        if self.left_context:
            state.history.copy_(x_in[..., safe_len : safe_len + self.left_context])
        state.pending = state.pending[..., safe_len:]
        return y

    def flush(self) -> torch.Tensor:
        state = self._streaming_state
        if state is None:
            raise RuntimeError("StreamingLookaheadConv1d.flush() requires streaming mode.")
        if state.pending.shape[-1] == 0:
            return state.pending.new_empty(state.batch_size, self.conv.out_channels, 0)  # type: ignore[attr-defined]

        right_pad = state.pending.new_zeros(state.batch_size, state.pending.shape[1], self.lookahead)
        x_in = torch.cat([state.history, state.pending, right_pad], dim=-1) if self.left_context else torch.cat([state.pending, right_pad], dim=-1)
        y = self.conv(x_in)
        if y.shape[-1] != state.pending.shape[-1]:
            raise RuntimeError(f"Internal flush length mismatch: expected {state.pending.shape[-1]}, got {y.shape[-1]}")
        state.reset()
        return y


@dataclass
class StreamingConvTranspose1dState(StreamingState):
    partial: torch.Tensor

    def reset(self) -> None:
        self.partial.zero_()


class StreamingConvTranspose1d(StreamingModule[StreamingConvTranspose1dState]):
    """Strict-causal ConvTranspose1d with a cached partial right tail."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size < stride:
            raise ValueError(f"kernel_size must be >= stride, got {kernel_size} < {stride}")
        self.convtr = _maybe_weight_norm(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=0,
                groups=groups,
                bias=bias,
            ),
            use_weight_norm,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.convtr.weight  # type: ignore[attr-defined]

    @property
    def bias(self) -> torch.Tensor | None:
        return self.convtr.bias  # type: ignore[attr-defined]

    @property
    def stride(self) -> int:
        return self.convtr.stride[0]  # type: ignore[attr-defined]

    @property
    def kernel_size(self) -> int:
        return self.convtr.kernel_size[0]  # type: ignore[attr-defined]

    @property
    def partial_size(self) -> int:
        return max(0, self.kernel_size - self.stride)

    def _init_streaming_state(self, batch_size: int) -> StreamingConvTranspose1dState:
        param = next(self.parameters())
        partial = torch.zeros(
            batch_size,
            self.convtr.out_channels,  # type: ignore[attr-defined]
            self.partial_size,
            dtype=param.dtype,
            device=param.device,
        )
        return StreamingConvTranspose1dState(batch_size, param.device, partial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")
        if x.shape[-1] == 0:
            return x.new_empty(x.shape[0], self.convtr.out_channels, 0)  # type: ignore[attr-defined]

        y = self.convtr(x)
        state = self._streaming_state
        if self.partial_size == 0:
            return y

        if state is None:
            return y[..., : -self.partial_size]

        y[..., : self.partial_size] += state.partial
        next_partial = y[..., -self.partial_size :]
        bias = self.convtr.bias  # type: ignore[attr-defined]
        if bias is not None:
            next_partial = next_partial - bias[:, None]
        state.partial.copy_(next_partial)
        return y[..., : -self.partial_size]


class StreamingAvgPool1d(StreamingModule[StreamingConv1dState]):
    """Causal average pooling implemented with a cached grouped convolution."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = int(stride)
        self.channels = int(channels)
        self.history_size = self.stride - 1
        weight = torch.full((channels, 1, stride), 1.0 / stride)
        self.register_buffer("weight", weight)

    def _init_streaming_state(self, batch_size: int) -> StreamingConv1dState:
        previous = torch.zeros(batch_size, self.channels, self.history_size, device=self.weight.device, dtype=self.weight.dtype)
        return StreamingConv1dState(batch_size, self.weight.device, previous)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.stride != 0:
            raise ValueError(
                f"StreamingAvgPool1d input length must be a multiple of stride={self.stride}, got {x.shape[-1]}"
            )
        state = self._streaming_state
        history = x.new_zeros(x.shape[0], x.shape[1], self.history_size) if state is None else state.previous
        x_in = torch.cat([history, x], dim=-1) if self.history_size else x
        y = F.conv1d(x_in, self.weight.to(dtype=x.dtype), stride=self.stride, groups=self.channels)
        if state is not None and self.history_size:
            state.previous.copy_(x_in[..., -self.history_size :])
        return y
