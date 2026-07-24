import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils import weight_norm

from .utils import init_weights, get_padding
from .backbones import ConvNeXtBlock, DownSamplingBlock


class Encoder(torch.nn.Module):
    def __init__(self, h):
        super(Encoder, self).__init__()

        self.input_channels = h.encoder_input_channels
        self.h = h
        self.in_dim = h.encoder_in_dim
        self.num_layers = h.encoder_num_layers
        self.out_dim = self.in_dim * (2 ** len(h.down_ratio))
        self.adanorm_num_embeddings = None
        self.layer_scale_init_value = [1 / num_layer for num_layer in self.num_layers]

        embed_ks = h.encoder_embed_kernel_size
        self.embed_mel = nn.Conv1d(self.input_channels, self.in_dim, kernel_size=embed_ks, padding=embed_ks // 2)
        self.norm_mel = nn.LayerNorm(self.in_dim, eps=1e-6)
        self.convnext_mel = nn.ModuleList()
        for i in range(len(self.num_layers)):
            for _ in range(self.num_layers[i]):
                cur_dim = self.in_dim * (2 ** i)
                cur_intermediate_dim = cur_dim * 4
                self.convnext_mel.append(
                    ConvNeXtBlock(cur_dim, cur_intermediate_dim, self.layer_scale_init_value[i], self.adanorm_num_embeddings)
                )
            self.convnext_mel.append(
                DownSamplingBlock(
                    cur_dim,
                    cur_dim * 2,
                    self.h.mel_Encoder_convnext_kernel_size[i],
                    self.h.down_ratio[i],
                    padding=(self.h.mel_Encoder_convnext_kernel_size[i] - self.h.down_ratio[i]) // 2,
                )
            )

        self.final_layer_norm_mel = nn.LayerNorm(self.out_dim, eps=1e-6)
        self.apply(self._init_weights)

        self.out_mel = torch.nn.Linear(self.out_dim, h.mel_Encoder_channel)

        self.mel_Encoder_output_conv = weight_norm(
            Conv1d(
                h.mel_Encoder_channel,
                h.mel_Encoder_channel // 4,
                h.mel_Encoder_output_kernel_size,
                1,
                padding=get_padding(h.mel_Encoder_output_kernel_size, 1),
            )
        )

        self.latent_output_conv = weight_norm(
            Conv1d(
                h.mel_Encoder_channel // 4,
                h.latent_dim,
                h.latent_output_conv_kernel_size,
                1,
                padding=get_padding(h.latent_output_conv_kernel_size, 1),
            )
        )

        self.mel_Encoder_output_conv.apply(init_weights)
        self.latent_output_conv.apply(init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, mel):
        mel_encode = self.embed_mel(mel)
        mel_encode = self.norm_mel(mel_encode.transpose(1, 2))
        mel_encode = mel_encode.transpose(1, 2)
        for conv_block in self.convnext_mel:
            mel_encode = conv_block(mel_encode)
        mel_encode = self.final_layer_norm_mel(mel_encode.transpose(1, 2))
        mel_encode = self.out_mel(mel_encode).transpose(1, 2)
        mel_encode = F.gelu(mel_encode)
        mel_encode = self.mel_Encoder_output_conv(mel_encode)
        mel_encode = F.gelu(mel_encode)
        latent = self.latent_output_conv(mel_encode)

        return latent
