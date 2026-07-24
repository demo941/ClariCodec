from __future__ import annotations

from dataclasses import dataclass

import torch

from src.codec.streaming import StreamingModule, StreamingState


@dataclass
class StreamingISTFTState(StreamingState):
    ola: torch.Tensor
    norm: torch.Tensor

    def reset(self) -> None:
        self.ola.zero_()
        self.norm.zero_()


class StreamingISTFT(StreamingModule[StreamingISTFTState]):
    """Causal overlap-add ISTFT.

    The non-streaming forward emits hop_length samples per spectrogram frame and
    drops the final unreleased tail. Streaming forward emits the same samples
    incrementally and keeps the overlap tail in state.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.register_buffer("window", torch.hann_window(win_length))

    def _init_streaming_state(self, batch_size: int) -> StreamingISTFTState:
        device = self.window.device
        ola = torch.zeros(batch_size, self.win_length, device=device, dtype=self.window.dtype)
        norm = torch.zeros_like(ola)
        return StreamingISTFTState(batch_size, device, ola, norm)

    def _frames(self, spec: torch.Tensor) -> torch.Tensor:
        frames = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        frames = frames[:, : self.win_length, :]
        return frames * self.window.to(frames.dtype)[None, :, None]

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.dim() != 3:
            raise ValueError(f"Expected complex spec [B, F, T], got {tuple(spec.shape)}")
        if spec.shape[-1] == 0:
            return spec.real.new_empty(spec.shape[0], 0)

        if self._streaming_state is not None:
            return self._forward_streaming(spec)
        return self._forward_full(spec)

    def _forward_full(self, spec: torch.Tensor) -> torch.Tensor:
        frames = self._frames(spec)
        B, _, T = frames.shape
        total = (T - 1) * self.hop_length + self.win_length
        y = frames.new_zeros(B, total)
        norm = frames.new_zeros(B, total)
        win_sq = self.window.to(frames.dtype).square()
        for t in range(T):
            start = t * self.hop_length
            end = start + self.win_length
            y[:, start:end] += frames[:, :, t]
            norm[:, start:end] += win_sq[None, :]
        y = y / norm.clamp_min(1e-8)
        return y[:, : T * self.hop_length]

    def _forward_streaming(self, spec: torch.Tensor) -> torch.Tensor:
        state = self._streaming_state
        assert state is not None
        frames = self._frames(spec)
        win_sq = self.window.to(frames.dtype).square()
        outs = []
        for t in range(frames.shape[-1]):
            state.ola[:, : self.win_length] += frames[:, :, t]
            state.norm[:, : self.win_length] += win_sq[None, :]
            out = state.ola[:, : self.hop_length] / state.norm[:, : self.hop_length].clamp_min(1e-8)
            outs.append(out)
            state.ola = torch.cat(
                [state.ola[:, self.hop_length :], state.ola.new_zeros(state.batch_size, self.hop_length)], dim=-1
            )
            state.norm = torch.cat(
                [state.norm[:, self.hop_length :], state.norm.new_zeros(state.batch_size, self.hop_length)], dim=-1
            )
        return torch.cat(outs, dim=-1)

    def flush(self) -> torch.Tensor:
        state = self._streaming_state
        if state is None:
            raise RuntimeError("StreamingISTFT.flush() requires streaming mode.")
        tail = state.ola / state.norm.clamp_min(1e-8)
        state.reset()
        return tail
