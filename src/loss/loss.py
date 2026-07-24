import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from ..vocos.modules import safe_log
import torchaudio

class GeneratorLoss(nn.Module):
    """
    Generator Loss module. Calculates the loss for the generator based on discriminator outputs.
    """

    def forward(self, disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            disc_outputs (List[Tensor]): List of discriminator outputs.

        Returns:
            Tuple[Tensor, List[Tensor]]: Tuple containing the total loss and a list of loss values from
                                         the sub-discriminators
        """
        loss = torch.zeros([], device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l

        return loss, gen_losses


class DiscriminatorLoss(nn.Module):
    """
    Discriminator Loss module. Calculates the loss for the discriminator based on real and generated outputs.
    """

    def forward(
        self, disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            disc_real_outputs (List[Tensor]): List of discriminator outputs for real samples.
            disc_generated_outputs (List[Tensor]): List of discriminator outputs for generated samples.

        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]: A tuple containing the total loss, a list of loss values from
                                                       the sub-discriminators for real outputs, and a list of
                                                       loss values for generated outputs.
        """
        loss = torch.zeros([], device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss)
            g_losses.append(g_loss)

        return loss, r_losses, g_losses


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss module. Calculates the feature matching loss between feature maps of the sub-discriminators.
    """

    def forward(self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            fmap_r (List[List[Tensor]]): List of feature maps from real samples.
            fmap_g (List[List[Tensor]]): List of feature maps from generated samples.

        Returns:
            Tensor: The calculated feature matching loss.
        """
        loss = torch.zeros([], device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss

def multi_scale_mel_loss(y, y_g):
    n_fft = 2048
    win_size = [128, 256, 512, 1024, 2048]
    hop_size = [32, 64, 128, 256, 512]
    sampling_rate = 16000
    num_mels = 80
    mel_losses = []

    if y.ndim == 3:
        y = y.squeeze(1)
        y_g = y_g.squeeze(1)

    for i in range(len(win_size)):
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_size[i],
            hop_length=hop_size[i],
            n_mels=num_mels,
            center=True,
            power=1,
        ).to(y.device)
        mel = safe_log(mel_spec(y))
        mel_g = safe_log(mel_spec(y_g))
        mel_loss = F.l1_loss(mel, mel_g)
        mel_losses.append(mel_loss)
    
    ms_mel_loss = sum(mel_losses) / len(mel_losses)
    
    return ms_mel_loss

def distillation_loss(ref_fea, syn_fea, lambda_sim=1.0):
    l1 = F.l1_loss(ref_fea, syn_fea, reduction='mean')
    cos = F.cosine_similarity(ref_fea, syn_fea, dim=-1).mean()
    return l1 - lambda_sim * torch.log(torch.sigmoid(cos))