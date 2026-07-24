from __future__ import annotations

import torch
from torch import nn

from src.codec.streaming import StreamingContainer
from src.vocos.streaming_spectral_ops import StreamingISTFT


class StreamingISTFTHead(StreamingContainer):
    def __init__(self, dim: int, n_fft: int, win_length: int, hop_length: int):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = StreamingISTFT(n_fft=n_fft, win_length=win_length, hop_length=hop_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag).clip(max=1e2)
        spec = mag * (torch.cos(p) + 1j * torch.sin(p))
        return self.istft(spec)

    def flush(self) -> torch.Tensor:
        return self.istft.flush()
