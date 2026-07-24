from typing import List, Optional, Tuple

import torch
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from torchaudio.transforms import Spectrogram


LRELU_SLOPE = 0.1

class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, h):
        super(MultiScaleDiscriminator, self).__init__()
        self.h = h
        self.discriminators = nn.ModuleList([
            DiscriminatorS(h),
            DiscriminatorS(h),
            DiscriminatorS(h),
        ])
        self.pooling = nn.AvgPool2d(kernel_size=4, stride=2, padding=1, count_include_pad=False)

    def forward(self, mel, mel_hat):
        mel_d_rs = []
        mel_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                mel = mel.unsqueeze(1)
                mel = self.pooling(mel)
                mel = mel.squeeze(1)

                mel_hat = mel_hat.unsqueeze(1)
                mel_hat = self.pooling(mel_hat)
                mel_hat = mel_hat.squeeze(1)

            mel_d_r, fmap_r = d(mel)
            mel_d_g, fmap_g = d(mel_hat)
            mel_d_rs.append(mel_d_r)
            fmap_rs.append(fmap_r)
            mel_d_gs.append(mel_d_g)
            fmap_gs.append(fmap_g)

        return mel_d_rs, mel_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(
        self,
        h,
        in_channels: int = 1,
        num_embeddings: int = None,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.h = h
        self.channels = h.msd_channels
        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList(
            [
                spectral_norm(nn.Conv2d(in_channels, self.channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
                spectral_norm(nn.Conv2d(self.channels, self.channels*2, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                spectral_norm(nn.Conv2d(self.channels*2, self.channels*4, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
                spectral_norm(nn.Conv2d(self.channels*4, self.channels*8, kernel_size=3, stride=(1, 1), padding=1)),
                spectral_norm(nn.Conv2d(self.channels*8, self.channels*16, kernel_size=3, stride=(1, 2), padding=1)),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.channels)
            torch.nn.init.zeros_(self.emb.weight)
        self.conv_post = spectral_norm(nn.Conv2d(self.channels*16, 1, (3, 3), padding=(1, 1)))

    def forward(
        self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None) :
        fmap = []
        x = x.unsqueeze(1)
        for layer in self.convs:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator module adapted from https://github.com/jik876/hifi-gan.
    Additionally, it allows incorporating conditional information with a learned embeddings table.

    Args:
        periods (tuple[int]): Tuple of periods for each discriminator.
        num_embeddings (int, optional): Number of embeddings. None means non-conditional discriminator.
            Defaults to None.
    """

    def __init__(self, h, periods: Tuple[int, ...] = (2, 3, 5, 7, 11), num_embeddings: Optional[int] = None):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(h=h, period=p, num_embeddings=num_embeddings) for p in periods])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        h,
        period: int,
        in_channels: int = 1,
        lrelu_slope: float = 0.1,
        num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.h = h
        self.period = period
        self.channels = h.mpd_channels
        self.kernel_size = h.mpd_kernel_size
        self.stride = h.mpd_stride
        self.convs = nn.ModuleList(
            [
                spectral_norm(Conv2d(in_channels, self.channels, (self.kernel_size, 1), (self.stride, 1), padding=(self.kernel_size // 2, 0))),
                spectral_norm(Conv2d(self.channels, self.channels*4, (self.kernel_size, 1), (self.stride, 1), padding=(self.kernel_size // 2, 0))),
                spectral_norm(Conv2d(self.channels*4, self.channels*16, (self.kernel_size, 1), (self.stride, 1), padding=(self.kernel_size // 2, 0))),
                spectral_norm(Conv2d(self.channels*16, self.channels*32, (self.kernel_size, 1), (self.stride, 1), padding=(self.kernel_size // 2, 0))),
                spectral_norm(Conv2d(self.channels*32, self.channels*32, (self.kernel_size, 1), (1, 1), padding=(self.kernel_size // 2, 0))),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=1024)
            torch.nn.init.zeros_(self.emb.weight)

        self.conv_post = spectral_norm(Conv2d(self.channels*32, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu_slope = lrelu_slope

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = x.unsqueeze(1)
        fmap = []
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for i, l in enumerate(self.convs):
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            if i > 0:
                fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        h,
        fft_sizes: Tuple[int, ...] = (2048, 1024, 512),
        num_embeddings: Optional[int] = None,
    ):
        """
        Multi-Resolution Discriminator module adapted from https://github.com/descriptinc/descript-audio-codec.
        Additionally, it allows incorporating conditional information with a learned embeddings table.

        Args:
            fft_sizes (tuple[int]): Tuple of window lengths for FFT. Defaults to (2048, 1024, 512).
            num_embeddings (int, optional): Number of embeddings. None means non-conditional discriminator.
                Defaults to None.
        """

        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(h=h,window_length=w, num_embeddings=num_embeddings) for w in fft_sizes]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: torch.Tensor = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(
        self,
        h,
        window_length: int,
        num_embeddings: Optional[int] = None,
        hop_factor: float = 0.25,
        bands: Tuple[Tuple[float, float], ...] = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)),
    ):
        super().__init__()
        self.h = h
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.channels = h.mrd_channels
        self.spec_fn = Spectrogram(
            n_fft=window_length, hop_length=int(window_length * hop_factor), win_length=window_length, power=None
        )
        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        convs = lambda: nn.ModuleList(
            [
                spectral_norm(nn.Conv2d(2, self.channels, (3, 9), (1, 1), padding=(1, 4))),
                spectral_norm(nn.Conv2d(self.channels, self.channels, (3, 9), (1, 2), padding=(1, 4))),
                spectral_norm(nn.Conv2d(self.channels, self.channels, (3, 9), (1, 2), padding=(1, 4))),
                spectral_norm(nn.Conv2d(self.channels, self.channels, (3, 9), (1, 2), padding=(1, 4))),
                spectral_norm(nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), padding=(1, 1))),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])

        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=channels)
            torch.nn.init.zeros_(self.emb.weight)

        self.conv_post = spectral_norm(nn.Conv2d(self.channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    def spectrogram(self, x):
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        x = self.spec_fn(x)
        x = torch.view_as_real(x)
        x = rearrange(x, "b f t c -> b c t f")
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None):
        x_bands = self.spectrogram(x)
        fmap = []
        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = layer(band)
                band = torch.nn.functional.leaky_relu(band, 0.1)
                if i > 0:
                    fmap.append(band)
            x.append(band)
        x = torch.cat(x, dim=-1)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h

        return x, fmap


class STFTDiscriminator(nn.Module):
    def __init__(self, filters, in_channels=2, out_channels=1):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(in_channels, filters, kernel_size=(3, 8), padding=(1, 3))),
            weight_norm(nn.Conv2d(filters, filters, kernel_size=(3, 8), stride=(2, 1), dilation=(1, 1), padding=(1, 3))),
            weight_norm(nn.Conv2d(filters, filters, kernel_size=(3, 8), stride=(2, 1), dilation=(1, 2), padding=(1, 3 + 1*2 // 2))),
            weight_norm(nn.Conv2d(filters, filters, kernel_size=(3, 8), stride=(2, 1), dilation=(1, 4), padding=(1, 3 + 3*4 // 2))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(filters, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

    def forward(self, x):
        fmap = []
        
        # x shape: [Batch, 2 (Real+Imag), Freq, Time]
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1) # Flatten for score calculation

        return x, fmap


class MultiScale_STFTDiscriminator(nn.Module):
    def __init__(self, filters=32):
        super().__init__()
        self.n_ffts = [2048, 1024, 512, 256, 128]
        self.hop_lengths = [n // 4 for n in self.n_ffts]
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(filters=filters, in_channels=2) 
            for _ in range(len(self.n_ffts))
        ])

    def forward(self, y, y_g):
        
        y = y.squeeze(1)
        y_g = y_g.squeeze(1)
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, (n_fft, hop_length) in enumerate(zip(self.n_ffts, self.hop_lengths)):
            y_stft = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft).to(y.device), return_complex=True, normalized=True)
            y_stft = torch.view_as_real(y_stft).permute(0, 3, 1, 2) 
            
            y_g_stft = torch.stft(y_g, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft).to(y_g.device), return_complex=True, normalized=True)
            y_g_stft = torch.view_as_real(y_g_stft).permute(0, 3, 1, 2)

            y_d_r, fmap_r = self.discriminators[i](y_stft)
            y_d_g, fmap_g = self.discriminators[i](y_g_stft)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs