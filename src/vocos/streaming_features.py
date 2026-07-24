from __future__ import annotations

import torch
import torchaudio
from torch import nn

from src.vocos.modules import safe_log


class CausalMelSpectrogramFeatures(nn.Module):
    """Causal log-mel extraction for full-sequence training.

    This uses center=False. For streaming inference, use StreamingMelSpectrogram
    below, which buffers waveform samples and emits new mel frames incrementally.
    """

    def __init__(self, sample_rate=16000, n_fft=1024, win_length=512, hop_length=160, n_mels=80):
        super().__init__()
        self.n_mels = int(n_mels)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False,
            power=1,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return safe_log(self.mel_spec(audio))


class CausalSingleMelLoss(nn.Module):
    """L1 log-mel loss using the same causal extraction as the streaming codec."""

    def __init__(self, sample_rate=16000, n_fft=1024, win_length=512, hop_length=160, n_mels=80):
        super().__init__()
        self.mel = CausalMelSpectrogramFeatures(sample_rate, n_fft, win_length, hop_length, n_mels)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mel_hat = self.mel(y_hat)
        mel = self.mel(y)
        frames = min(mel_hat.shape[-1], mel.shape[-1])
        return torch.nn.functional.l1_loss(mel_hat[..., :frames], mel[..., :frames])


class StreamingMelSpectrogram(nn.Module):
    """Stateful causal mel extractor.

    The caller can push arbitrary sample chunks. The module keeps incomplete
    waveform history and returns all newly available mel frames.
    """

    def __init__(self, sample_rate=16000, n_fft=1024, win_length=512, hop_length=160, n_mels=80):
        super().__init__()
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.mel = CausalMelSpectrogramFeatures(sample_rate, n_fft, win_length, hop_length, n_mels)
        self._buffer: torch.Tensor | None = None

    def reset(self) -> None:
        self._buffer = None

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() != 2:
            raise ValueError(f"Expected audio [B, T], got {tuple(audio.shape)}")
        if self._buffer is None:
            self._buffer = audio
        else:
            self._buffer = torch.cat([self._buffer, audio], dim=-1)

        if self._buffer.shape[-1] < self.n_fft:
            return audio.new_empty(audio.shape[0], self.n_mels, 0)

        num_frames = (self._buffer.shape[-1] - self.n_fft) // self.hop_length + 1
        usable = (num_frames - 1) * self.hop_length + self.n_fft
        chunk = self._buffer[..., :usable]
        features = self.mel(chunk)

        keep_from = num_frames * self.hop_length
        self._buffer = self._buffer[..., keep_from:]
        return features

    def flush(self) -> torch.Tensor:
        if self._buffer is None:
            raise RuntimeError("StreamingMelSpectrogram.flush() called before forward().")
        if self._buffer.shape[-1] == 0:
            return self._buffer.new_empty(self._buffer.shape[0], self.n_mels, 0)
        pad = max(0, self.n_fft - self._buffer.shape[-1])
        audio = torch.nn.functional.pad(self._buffer, (0, pad))
        self._buffer = None
        return self.mel(audio)


def pad_mel_to_multiple(mel: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    remainder = mel.shape[-1] % multiple
    if remainder == 0:
        return mel, 0
    pad = multiple - remainder
    return torch.nn.functional.pad(mel, (0, pad)), pad
