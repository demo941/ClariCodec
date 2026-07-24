from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .streaming import StreamingContainer
from .streaming_backbones import StreamingConvNeXtBlock, StreamingUpSamplingBlock
from .streaming_conv import StreamingConv1d
from .utils import init_weights


class StreamingDecoder(StreamingContainer):
    def __init__(self, h):
        super().__init__()
        self.h = h
        base_dim = h.decoder_base_dim
        self.in_dim = base_dim * (2 ** len(h.up_ratio))
        self.out_dim = base_dim
        self.num_layers = h.decoder_num_layers
        self.layer_scale_init_value = [1 / num_layer for num_layer in self.num_layers]
        self.output_channels = h.decoder_output_channels
        self.use_sampling_shortcut = bool(h.streaming_sampling_shortcut)

        self.latent_input_conv = StreamingConv1d(
            h.latent_dim,
            h.mel_Decoder_channel // 4,
            kernel_size=h.latent_input_conv_kernel_size,
            use_weight_norm=True,
        )
        self.mel_Decoder_input_conv = StreamingConv1d(
            h.mel_Decoder_channel // 4,
            h.mel_Decoder_channel,
            kernel_size=h.mel_Decoder_input_kernel_size,
            use_weight_norm=True,
        )
        self.in_mel = nn.Linear(h.mel_Decoder_channel, self.in_dim)
        self.norm_mel = nn.LayerNorm(self.in_dim, eps=1e-6)
        self.convnext_mel = nn.ModuleList()
        for i in range(len(self.num_layers)):
            cur_dim = self.in_dim // (2 ** i)
            cur_intermediate_dim = cur_dim * 2
            self.convnext_mel.append(
                StreamingUpSamplingBlock(
                    cur_dim,
                    cur_dim // 2,
                    self.h.mel_Decoder_convnext_kernel_size[i],
                    self.h.up_ratio[i],
                    use_shortcut=self.use_sampling_shortcut,
                )
            )
            for _ in range(self.num_layers[i]):
                self.convnext_mel.append(
                    StreamingConvNeXtBlock(cur_dim // 2, cur_intermediate_dim, self.layer_scale_init_value[i])
                )

        self.final_layer_norm_mel = nn.LayerNorm(self.out_dim, eps=1e-6)
        self.mel_Decoder_output_conv = StreamingConv1d(
            self.out_dim,
            h.decoder_output_channels,
            kernel_size=h.mel_Decoder_output_conv_kernel_size,
            use_weight_norm=True,
        )
        self.apply(self._init_weights)
        self.mel_Decoder_output_conv.apply(init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = F.gelu(latent)
        x = self.latent_input_conv(x)
        x = F.gelu(x)
        x = self.mel_Decoder_input_conv(x)
        x = F.gelu(x)
        x = self.in_mel(x.transpose(1, 2))
        x = self.norm_mel(x).transpose(1, 2)
        for block in self.convnext_mel:
            x = block(x)
        x = self.final_layer_norm_mel(x.transpose(1, 2)).transpose(1, 2)
        return self.mel_Decoder_output_conv(x)
