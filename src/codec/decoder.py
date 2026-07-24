import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils import weight_norm

from .utils import init_weights, get_padding
from .backbones import ConvNeXtBlock, UpSamplingBlock

class Decoder(torch.nn.Module):
    def __init__(self, h):
        super(Decoder, self).__init__()

        self.h=h
        base_dim = h.decoder_base_dim
        self.in_dim = base_dim * (2**len(h.up_ratio))
        self.out_dim = base_dim
        self.num_layers = h.decoder_num_layers
        self.adanorm_num_embeddings=None
        self.layer_scale_init_value =  [1/num_layer for num_layer in self.num_layers]
        self.output_channels = h.decoder_output_channels

        self.latent_input_conv = weight_norm(Conv1d(h.latent_dim, h.mel_Decoder_channel//4, h.latent_input_conv_kernel_size, 1, 
                                                padding=get_padding(h.latent_input_conv_kernel_size, 1)))

        self.mel_Decoder_input_conv = weight_norm(Conv1d(h.mel_Decoder_channel//4, h.mel_Decoder_channel, h.mel_Decoder_input_kernel_size, 1, 
                                                padding=get_padding(h.mel_Decoder_input_kernel_size, 1)))

        self.in_mel = torch.nn.Linear(h.mel_Decoder_channel, self.in_dim)
        self.norm_mel = nn.LayerNorm(self.in_dim, eps=1e-6)
        self.convnext_mel = nn.ModuleList()
        for i in range(len(self.num_layers)):
            cur_dim = self.in_dim // (2**i)
            cur_intermediate_dim = cur_dim * 2
            self.convnext_mel.append(UpSamplingBlock(cur_dim, cur_dim//2, self.h.mel_Decoder_convnext_kernel_size[i], self.h.up_ratio[i],
                                                  padding=(self.h.mel_Decoder_convnext_kernel_size[i] - self.h.up_ratio[i]) // 2))
            for _ in range(self.num_layers[i]):
                self.convnext_mel.append(ConvNeXtBlock(cur_dim//2, cur_intermediate_dim, self.layer_scale_init_value[i], self.adanorm_num_embeddings))
        
        self.final_layer_norm_mel = nn.LayerNorm(self.out_dim, eps=1e-6)
        self.apply(self._init_weights)

        self.mel_Decoder_output_conv = weight_norm(Conv1d(self.out_dim, h.decoder_output_channels, h.mel_Decoder_output_conv_kernel_size, 1, 
                                                  padding=get_padding(h.mel_Decoder_output_conv_kernel_size, 1)))

        self.mel_Decoder_output_conv.apply(init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, latent):
        #input: [batch_size, latent_dim, frames//ratio]
        latent = F.gelu(latent)
        latent = self.latent_input_conv(latent)
        latent = F.gelu(latent)
        mel = self.mel_Decoder_input_conv(latent)
        mel = F.gelu(mel)
        mel = self.in_mel(mel.transpose(1, 2))
        mel = self.norm_mel(mel)
        mel = mel.transpose(1, 2)
        for conv_block in self.convnext_mel:
            mel = conv_block(mel)
        mel = self.final_layer_norm_mel(mel.transpose(1, 2))
        mel = mel.transpose(1, 2)
        mel = self.mel_Decoder_output_conv(mel) #output: [batch_size, h.num_mels, frames]

        return mel
