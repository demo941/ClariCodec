from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

import librosa
import soundfile as sf
import torch
from tqdm import tqdm

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.codec.quantizer import build_quantizer
from src.codec.streaming_decoder import StreamingDecoder
from src.codec.streaming_encoder import StreamingEncoder
from src.codec.utils import (
    build_stats_accumulator as _build_stats_accumulator,
    compute_file_bits as _compute_file_bits,
    get_codebook_sizes as _get_codebook_sizes,
    load_checkpoint,
    load_hparams,
    set_quantizer_mode as _set_quantizer_mode,
)
from src.vocos.streaming_features import StreamingMelSpectrogram, pad_mel_to_multiple
from src.vocos.streaming_heads import StreamingISTFTHead
from src.vocos.streaming_models import StreamingVocosBackbone


device = None
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@torch.no_grad()
def _decode_latent_frames(latent, quantizer, decoder, vocosbackbone, istfthead):
    if latent.shape[-1] == 0:
        return latent.new_empty(latent.shape[0], 0), None
    quantizer_out = quantizer(latent)
    mel_latent = decoder(quantizer_out.z_q)
    vocoder_emb = vocosbackbone(mel_latent)
    audio = istfthead(vocoder_emb)
    return audio, quantizer_out.codes


@torch.no_grad()
def stream_file(raw_wav: torch.Tensor, h, encoder, quantizer, decoder, vocosbackbone, istfthead):
    frame_ratio = math.prod(h.down_ratio)
    mel_stream = StreamingMelSpectrogram(
        sample_rate=h.sampling_rate,
        n_fft=h.n_fft,
        win_length=h.win_size,
        hop_length=h.hop_size,
        n_mels=h.num_mels,
    ).to(device)

    stream_samples = int(h.stream_audio_chunk_samples)
    pending_mel = None
    audio_parts = []
    code_parts = []
    tail_silence_samples = 0

    with encoder.streaming(1), decoder.streaming(1), vocosbackbone.streaming(1), istfthead.streaming(1):
        for start in range(0, raw_wav.shape[-1], stream_samples):
            audio_chunk = raw_wav[:, start : start + stream_samples]
            mel = mel_stream(audio_chunk)
            if pending_mel is not None:
                mel = torch.cat([pending_mel, mel], dim=-1)
            usable = (mel.shape[-1] // frame_ratio) * frame_ratio
            if usable > 0:
                latent = encoder(mel[..., :usable])
                audio, codes = _decode_latent_frames(latent, quantizer, decoder, vocosbackbone, istfthead)
                audio_parts.append(audio)
                if codes is not None:
                    code_parts.append(codes)
            pending_mel = mel[..., usable:]

        mel_flush_padding_samples = 0
        if mel_stream._buffer is not None and mel_stream._buffer.shape[-1] > 0:
            mel_flush_padding_samples = max(0, h.n_fft - mel_stream._buffer.shape[-1])
        tail_mel = mel_stream.flush()
        if pending_mel is not None:
            tail_mel = torch.cat([pending_mel, tail_mel], dim=-1)
        if tail_mel.shape[-1] > 0:
            tail_mel, padded_mel_frames = pad_mel_to_multiple(tail_mel, frame_ratio)
            if padded_mel_frames > 0:
                tail_silence_samples = max(tail_silence_samples, frame_ratio * h.hop_size)
            tail_silence_samples = max(tail_silence_samples, mel_flush_padding_samples)
            latent = encoder(tail_mel)
            audio, codes = _decode_latent_frames(latent, quantizer, decoder, vocosbackbone, istfthead)
            audio_parts.append(audio)
            if codes is not None:
                code_parts.append(codes)
        latent_tail = encoder.flush()
        if latent_tail.shape[-1] > 0:
            audio, codes = _decode_latent_frames(latent_tail, quantizer, decoder, vocosbackbone, istfthead)
            audio_parts.append(audio)
            if codes is not None:
                code_parts.append(codes)
        audio_parts.append(istfthead.flush())

    if audio_parts:
        audio = torch.cat(audio_parts, dim=-1)
    else:
        audio = raw_wav.new_empty(1, 0)
    if audio.shape[-1] > raw_wav.shape[-1]:
        audio = audio[..., : raw_wav.shape[-1]]
    elif audio.shape[-1] < raw_wav.shape[-1]:
        audio = torch.nn.functional.pad(audio, (0, raw_wav.shape[-1] - audio.shape[-1]))
    # Finite-file evaluation cleanup only: these samples come from flushing/padding
    # an artificially ended utterance. A real always-on streaming system would not
    # transmit this side information or spend extra bitrate on it.
    if tail_silence_samples > 0 and audio.shape[-1] > 0:
        tail_silence_samples = min(tail_silence_samples, audio.shape[-1])
        audio = audio.clone()
        audio[..., -tail_silence_samples:] = 0.0

    codes = torch.cat(code_parts, dim=-1) if code_parts else None
    return audio, codes


def inference(h):
    encoder = StreamingEncoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    decoder = StreamingDecoder(h).to(device)
    vocosbackbone = StreamingVocosBackbone(
        input_channels=h.vocos_backbone_input_channels,
        dim=h.vocos_backbone_dim,
        intermediate_dim=h.vocos_backbone_intermediate_dim,
        num_layers=h.vocos_backbone_num_layers,
    ).to(device)
    istfthead = StreamingISTFTHead(
        dim=h.vocos_head_dim,
        n_fft=h.vocos_head_n_fft,
        win_length=h.vocos_head_win_length,
        hop_length=h.vocos_head_hop_length,
    ).to(device)

    state_dict_codec = load_checkpoint(h.checkpoint_file_load_Codec, device)
    encoder.load_state_dict(state_dict_codec["encoder"], strict=True)
    quantizer.load_state_dict(state_dict_codec["quantizer"], strict=True)
    decoder.load_state_dict(state_dict_codec["decoder"], strict=True)
    state_dict_vocos = load_checkpoint(h.checkpoint_file_load_Vocos, device)
    vocosbackbone.load_state_dict(state_dict_vocos["vocosbackbone"], strict=True)
    istfthead.load_state_dict(state_dict_vocos["istfthead"], strict=True)

    for m in (encoder, quantizer, decoder, vocosbackbone, istfthead):
        m.eval()
    _set_quantizer_mode(quantizer, stochastic=False, temperature=0.3)

    codebook_sizes = _get_codebook_sizes(quantizer)
    stats = _build_stats_accumulator(codebook_sizes)
    total_duration = 0.0
    total_bits = 0
    os.makedirs(h.test_wav_output_dir, exist_ok=True)

    filelist = sorted(os.listdir(h.test_input_wavs_dir))
    start_time = time.time()
    for filename in tqdm(filelist, desc="streaming inference"):
        input_path = os.path.join(h.test_input_wavs_dir, filename)
        if not os.path.isfile(input_path):
            continue
        wav, sr = librosa.load(input_path, sr=h.sampling_rate, mono=True)
        total_duration += len(wav) / sr
        raw_wav = torch.tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)
        audio, codes = stream_file(raw_wav, h, encoder, quantizer, decoder, vocosbackbone, istfthead)
        if stats is not None and codes is not None:
            stats.update(codes)
        total_bits += _compute_file_bits(codes, codebook_sizes)
        audio_name = os.path.splitext(filename)[0]
        sf.write(
            os.path.join(h.test_wav_output_dir, audio_name + ".wav"),
            audio.squeeze(0).cpu().numpy(),
            h.sampling_rate,
            "PCM_16",
        )

    elapsed = time.time() - start_time
    bitrate_kbps = (total_bits / total_duration) / 1000.0 if total_duration > 0 else 0.0
    logging.info("Total streaming inference time: %.2f seconds", elapsed)
    logging.info("Total duration: %.3f seconds, Total bits: %d, Bitrate: %.3f kbps", total_duration, total_bits, bitrate_kbps)
    if stats is not None:
        print(json.dumps(stats.compute(), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to config YAML")
    args = parser.parse_args()
    h = load_hparams(args.config)

    torch.manual_seed(h.seed)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    inference(h)


if __name__ == "__main__":
    main()
