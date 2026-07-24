from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .streaming import StreamingContainer
from .streaming_backbones import StreamingConvNeXtBlock, StreamingDownSamplingBlock
from .streaming_conv import StreamingConv1d, StreamingLookaheadConv1d
from .utils import init_weights


class StreamingEncoder(StreamingContainer):
    def __init__(self, h):
        super().__init__()
        self.input_channels = h.encoder_input_channels
        self.h = h
        self.latent_dim = h.latent_dim
        self.in_dim = h.encoder_in_dim
        self.num_layers = h.encoder_num_layers
        self.out_dim = self.in_dim * (2 ** len(h.down_ratio))
        self.layer_scale_init_value = [1 / num_layer for num_layer in self.num_layers]
        self.use_sampling_shortcut = bool(h.streaming_sampling_shortcut)
        self.frame_ratio = 1
        for ratio in h.down_ratio:
            self.frame_ratio *= int(ratio)

        embed_ks = h.encoder_embed_kernel_size
        self.embed_mel = StreamingConv1d(self.input_channels, self.in_dim, kernel_size=embed_ks)
        self.norm_mel = nn.LayerNorm(self.in_dim, eps=1e-6)
        self.convnext_mel = nn.ModuleList()
        for i in range(len(self.num_layers)):
            for _ in range(self.num_layers[i]):
                cur_dim = self.in_dim * (2 ** i)
                cur_intermediate_dim = cur_dim * 4
                self.convnext_mel.append(
                    StreamingConvNeXtBlock(cur_dim, cur_intermediate_dim, self.layer_scale_init_value[i])
                )
            self.convnext_mel.append(
                StreamingDownSamplingBlock(
                    cur_dim,
                    cur_dim * 2,
                    self.h.mel_Encoder_convnext_kernel_size[i],
                    self.h.down_ratio[i],
                    use_shortcut=self.use_sampling_shortcut,
                )
            )

        self.final_layer_norm_mel = nn.LayerNorm(self.out_dim, eps=1e-6)
        self.out_mel = nn.Linear(self.out_dim, h.mel_Encoder_channel)
        self.mel_Encoder_output_conv = StreamingConv1d(
            h.mel_Encoder_channel,
            h.mel_Encoder_channel // 4,
            kernel_size=h.mel_Encoder_output_kernel_size,
            use_weight_norm=True,
        )
        self.latent_output_conv = StreamingConv1d(
            h.mel_Encoder_channel // 4,
            h.latent_dim,
            kernel_size=h.latent_output_conv_kernel_size,
            use_weight_norm=True,
        )
        latent_lookahead_frames = h.latent_lookahead_frames
        if latent_lookahead_frames > 0:
            latent_lookahead_kernel_size = h.latent_lookahead_kernel_size
            self.latent_lookahead = StreamingLookaheadConv1d(
                h.latent_dim,
                kernel_size=latent_lookahead_kernel_size,
                lookahead=latent_lookahead_frames,
                use_weight_norm=True,
            )
        else:
            self.latent_lookahead = None
        self.apply(self._init_weights)
        self.mel_Encoder_output_conv.apply(init_weights)
        self.latent_output_conv.apply(init_weights)
        if self.latent_lookahead is not None:
            self.latent_lookahead.apply(init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.shape[-1] % self.frame_ratio != 0:
            raise ValueError(
                f"StreamingEncoder expects mel frames to be a multiple of down_ratio={self.frame_ratio}, "
                f"got {mel.shape[-1]}"
            )
        x = self.embed_mel(mel)
        x = self.norm_mel(x.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext_mel:
            x = block(x)
        x = self.final_layer_norm_mel(x.transpose(1, 2))
        x = self.out_mel(x).transpose(1, 2)
        x = F.gelu(x)
        x = self.mel_Encoder_output_conv(x)
        x = F.gelu(x)
        x = self.latent_output_conv(x)
        if self.latent_lookahead is not None:
            x = self.latent_lookahead(x)
        return x

    def flush(self) -> torch.Tensor:
        if self.latent_lookahead is not None:
            return self.latent_lookahead.flush()
        param = next(self.parameters())
        batch_size = self._streaming_state.batch_size if self._streaming_state is not None else 0
        return param.new_empty(batch_size, self.latent_dim, 0)
