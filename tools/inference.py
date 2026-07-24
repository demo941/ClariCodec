from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import argparse
import json
import logging
import time

import librosa
import soundfile as sf
import torch

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.codec.utils import (
    build_stats_accumulator as _build_stats_accumulator,
    compute_file_bits as _compute_file_bits,
    get_codebook_sizes as _get_codebook_sizes,
    load_hparams,
    load_checkpoint,
    set_quantizer_mode as _set_quantizer_mode,
)
from src.codec.encoder_wo_quantize import Encoder
from src.codec.decoder import Decoder
from src.codec.quantizer import build_quantizer
from src.vocos.feature_extractors import MelSpectrogramFeatures
from src.vocos.heads import ISTFTHead
from src.vocos.models import VocosBackbone


device = None
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def inference(h):
    encoder = Encoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    decoder = Decoder(h).to(device)
    vocosbackbone = VocosBackbone(
        input_channels=h.vocos_backbone_input_channels,
        dim=h.vocos_backbone_dim,
        intermediate_dim=h.vocos_backbone_intermediate_dim,
        num_layers=h.vocos_backbone_num_layers,
    ).to(device)
    istfthead = ISTFTHead(
        dim=h.vocos_head_dim,
        n_fft=h.vocos_head_n_fft,
        win_length=h.vocos_head_win_length,
        hop_length=h.vocos_head_hop_length,
        padding=h.vocos_head_padding,
    ).to(device)
    mel_spectrogram = MelSpectrogramFeatures(
        sample_rate=h.sampling_rate,
        n_fft=h.n_fft,
        win_length=h.win_size,
        hop_length=h.hop_size,
        n_mels=h.num_mels,
        padding=h.mel_padding,
    ).to(device)

    state_dict_codec = load_checkpoint(h.checkpoint_file_load_Codec, device)
    encoder.load_state_dict(state_dict_codec["encoder"], strict=True)
    quantizer.load_state_dict(state_dict_codec["quantizer"], strict=True)
    decoder.load_state_dict(state_dict_codec["decoder"], strict=True)

    state_dict_vocos = load_checkpoint(h.checkpoint_file_load_Vocos, device)
    vocosbackbone.load_state_dict(state_dict_vocos["vocosbackbone"], strict=True)
    istfthead.load_state_dict(state_dict_vocos["istfthead"], strict=True)

    filelist = sorted(os.listdir(h.test_input_wavs_dir))

    os.makedirs(h.test_wav_output_dir, exist_ok=True)
    os.makedirs(h.vis_output_dir, exist_ok=True)

    encoder.eval()
    quantizer.eval()
    decoder.eval()
    vocosbackbone.eval()
    istfthead.eval()
    _set_quantizer_mode(quantizer, stochastic=False, temperature=0.3)

    codebook_sizes = _get_codebook_sizes(quantizer)
    stats = _build_stats_accumulator(codebook_sizes)

    total_duration = 0.0
    total_bits = 0

    with torch.no_grad():
        start_time = time.time()
        for filename in filelist:
            input_path = os.path.join(h.test_input_wavs_dir, filename)
            if not os.path.isfile(input_path):
                continue

            raw_wav, sr = librosa.load(input_path, sr=h.sampling_rate, mono=True)
            duration = len(raw_wav) / sr
            total_duration += duration

            raw_wav = torch.FloatTensor(raw_wav).to(device)
            mel = mel_spectrogram(raw_wav.unsqueeze(0))

            latent = encoder(mel)
            quantizer_out = quantizer(latent)
            codes = quantizer_out.codes

            if stats is not None and codes is not None:
                stats.update(codes)
            total_bits += _compute_file_bits(codes, codebook_sizes)

            mel_latent = decoder(quantizer_out.z_q)
            vocoder_emb = vocosbackbone(mel_latent)
            y_g = istfthead(vocoder_emb)
            mel_g = mel_spectrogram(y_g)

            audio = y_g.squeeze()
            audio_name = os.path.splitext(filename)[0]

            sf.write(
                os.path.join(h.test_wav_output_dir, audio_name + ".wav"),
                audio.cpu().numpy(),
                h.sampling_rate,
                "PCM_16",
            )

        total_inference_time = time.time() - start_time

    bitrate_kbps = 0.0
    if total_duration > 0:
        bitrate_kbps = (total_bits / total_duration) / 1000.0

    logging.info(f"Total inference time: {total_inference_time:.2f} seconds")
    logging.info(f"Total duration: {total_duration:.3f} seconds, Total bits: {total_bits}, Bitrate: {bitrate_kbps:.3f} kbps")

    if stats is not None:
        print(json.dumps(stats.compute(), indent=2))


def main():
    print("Initializing Inference Process..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to config YAML")
    args = parser.parse_args()
    h = load_hparams(args.config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inference(h)


if __name__ == "__main__":
    main()
