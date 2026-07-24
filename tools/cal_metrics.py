import os
import sys
import argparse
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from rich.progress import track
from pystoi.stoi import stoi
from pesq import pesq
from typing import Optional
import csv

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.codec.utils import load_hparams

LOG_FILE_PATH: Optional[str] = None
TARGET_SAMPLE_RATE = 16000

def set_log_file_path(path: Optional[str]):
    global LOG_FILE_PATH
    LOG_FILE_PATH = path

    if LOG_FILE_PATH:
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

def write_log(msg: str):
    """Print to the console and append to the log file."""
    print(msg)
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def stft(audio, n_fft=2048, hop_length=512):
    hann_window = torch.hann_window(n_fft).to(audio.device)
    stft_spec = torch.stft(audio, n_fft, hop_length, window=hann_window, return_complex=True)
    stft_mag = torch.abs(stft_spec)
    stft_pha = torch.angle(stft_spec)

    return stft_mag, stft_pha

def cal_stoi_score(pred, target, sr):
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    return stoi(target_np, pred_np, sr, extended=False)

def cal_pesq_score(pred, target, sr):
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    if sr == 16000:
        return pesq(16000, target_np, pred_np, "wb", on_error=1)
    else:
        raise ValueError("PESQ only supports 16000Hz sampling rates.")

def resample_to_target_sr(wav, sr, target_sr=TARGET_SAMPLE_RATE):
    if sr == target_sr:
        return wav, sr
    resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
    return resampler(wav), target_sr

def main(h):

    wav_indexes = os.listdir(h.test_input_wavs_dir)
    csv_file_handle = open(h.metrics_detailed_log_path, "w", encoding="utf-8-sig", newline="")
    writer = csv.writer(csv_file_handle)
    writer.writerow(["Filename", "STOI", "PESQ"])
    csv_file_handle.flush()
    
    metrics = {
        'stoi':[], 'pesq':[]
    }

    for wav_index in track(wav_indexes):

        ref_path = os.path.join(h.test_input_wavs_dir, wav_index)
        syn_path = os.path.join(h.test_wav_output_dir, wav_index)

        ref_wav, ref_sr = torchaudio.load(ref_path)
        syn_wav, syn_sr = torchaudio.load(syn_path)
        ref_wav, ref_sr = resample_to_target_sr(ref_wav, ref_sr)
        syn_wav, syn_sr = resample_to_target_sr(syn_wav, syn_sr)

        length = min(ref_wav.size(1), syn_wav.size(1))
        ref_wav = ref_wav[:, : length].to(device)
        syn_wav = syn_wav[:, : length].to(device)

        stoi_score = cal_stoi_score(syn_wav, ref_wav, sr=TARGET_SAMPLE_RATE)
        pesq_score = cal_pesq_score(syn_wav, ref_wav, sr=TARGET_SAMPLE_RATE)
        writer.writerow([
            os.path.basename(wav_index), f"{stoi_score:.3f}", f"{pesq_score:.3f}"
        ])

        metrics['stoi'].append(torch.tensor(stoi_score))
        metrics['pesq'].append(torch.tensor(pesq_score))

    write_log('STOI: {:.3f}'.format(torch.stack(metrics['stoi']).mean()))
    write_log('PESQ: {:.3f}'.format(torch.stack(metrics['pesq']).mean()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to config YAML")
    args = parser.parse_args()
    config_file = args.config
    h = load_hparams(config_file)
    set_log_file_path(h.metrics_log_file_path)

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    main(h)
