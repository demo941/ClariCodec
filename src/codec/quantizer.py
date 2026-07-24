from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from .stochastic_fsq import RFSQ


@dataclass
class QuantizerOutput:
    z_q: torch.Tensor
    codes: Optional[torch.Tensor]
    log_probs: Optional[Any] = None



class BaseQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_stochastic_quantizer = False

    @property
    def codebook_sizes(self) -> Tuple[int, ...]:
        return ()

    def set_stochastic_mode(self, stochastic: bool, temperature: float) -> None:
        _ = stochastic
        _ = temperature

    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        raise NotImplementedError


class RFSQQuantizer(BaseQuantizer):
    def __init__(self, h):
        super().__init__()
        latent_dim = int(h.latent_dim)
        num_quantizers = int(h.num_quantizers)
        levels = h.levels
        use_stochastic = bool(h.stochastic)

        self.rfsq = RFSQ(
            levels=levels,
            dim=latent_dim,
            num_quantizers=num_quantizers,
            channel_first=True,
            stochastic=use_stochastic,
        )
        self._num_quantizers = num_quantizers
        self._codebook_size = int(self.rfsq.codebook_size)
        self.is_stochastic_quantizer = True

    @property
    def codebook_sizes(self) -> Tuple[int, ...]:
        return tuple([self._codebook_size] * self._num_quantizers)

    def set_stochastic_mode(self, stochastic: bool, temperature: float) -> None:
        self.rfsq.set_stochastic(stochastic)
        self.rfsq.set_temperature(temperature)

    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        z_q, codes = self.rfsq(latent)
        return QuantizerOutput(z_q=z_q, codes=codes, log_probs=self.rfsq.saved_log_probs)


class NoQuantizer(BaseQuantizer):
    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        return QuantizerOutput(z_q=latent, codes=None)


def build_quantizer(h) -> BaseQuantizer:
    quantizer_type = str(h.quantizer_type).lower()
    if quantizer_type == "rfsq":
        return RFSQQuantizer(h)
    if quantizer_type == "none":
        return NoQuantizer()
    raise ValueError(f"Unsupported quantizer_type: {quantizer_type}. Expected one of: rfsq, none.")
