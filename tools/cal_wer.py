import os
import sys
import glob
import re
import argparse
import csv
import logging
from copy import deepcopy
from typing import List, Optional

import torch
import soundfile as sf
from jiwer import wer, process_words
from tqdm import tqdm

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.codec.utils import load_hparams
from omegaconf import open_dict
from whisper.normalizers import EnglishTextNormalizer

# ================= Configuration =================

NEMO_MODEL_PATH = "your_asr_model_path"

# Key setting: dynamic batch control.
# GPU capacity factor = audio duration in seconds * batch size.
# If the longest sample is 35s and batch size 4 fits, the factor is 35 * 4 = 140.
# Use a smaller factor for smaller GPUs, and a larger factor for 40G/80G GPUs.
GPU_CAPACITY_FACTOR = 8000

# Do not exceed this batch size even for short audio, due to CPU preprocessing overhead.
MAX_BATCH_SIZE_LIMIT = 256

# Path settings.
LOG_FILE_PATH: Optional[str] = None
PRINT_TARGETS = {"121-121726-0000.wav"}

# ===========================================

logging.getLogger("nemo_logger").setLevel(logging.ERROR)

def set_log_file_path(path: Optional[str]):
    global LOG_FILE_PATH
    if path:
        LOG_FILE_PATH = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

def write_log(msg: str):
    print(msg)
    if LOG_FILE_PATH:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

def normalize_text(text) -> str:
    # Support NeMo RNNT Hypothesis objects.
    if hasattr(text, "text") and isinstance(getattr(text, "text"), str):
        text = text.text

    if not isinstance(text, str):
        return ""

    # text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # text = text.upper()
    # text = " ".join(text.split())
    return text

def load_nemo_model(model_path: str):
    import nemo.collections.asr as nemo_asr
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    print(f"Loading NeMo model from: {model_path} ...")
    try:
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=model_path)
    except Exception:
        asr_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
    if torch.cuda.is_available():
        asr_model = asr_model.cuda()
    asr_model.eval()
    return asr_model

def load_ref_texts(ref_path: str) -> dict:
    utt2txt = {}
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference text file not found: {ref_path}")

    with open(ref_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(.*)\s+\(([^()]+)\)$", line)
            if match is None:
                raise ValueError(
                    f"Invalid ref line format in {ref_path}: {line}. "
                    "Expected: reference text (utt_id)"
                )
            text, utt_id = match.group(1).strip(), match.group(2).strip()
            if text and utt_id:
                utt2txt[utt_id] = text

    if not utt2txt:
        raise ValueError(f"No valid reference texts found in {ref_path}")
    return utt2txt

def get_audio_duration(wav_path: str) -> float:
    """
    Read the audio header to get duration without decoding the full waveform.
    """
    try:
        info = sf.info(wav_path)  # Header only.
        duration = float(info.frames) / float(info.samplerate)
        return max(0.01, duration)
    except Exception:
        return 10.0 # Default fallback.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to config YAML")
    parser.add_argument("--model_path", type=str, default=NEMO_MODEL_PATH)
    parser.add_argument("--capacity", type=int, default=GPU_CAPACITY_FACTOR, help="GPU Capacity Factor (Secs * Batch)")
    args = parser.parse_args()
    config_file = args.config
    h = load_hparams(config_file)
    wav_dir = h.test_wav_output_dir
    set_log_file_path(h.wer_log_file_path)
    detailed_log_path = h.wer_detailed_file_path

    ref_path = h.ref_path
    hyp_path = h.hyp_path
    print(f"Detailed CSV: {detailed_log_path}")
    ref_texts = load_ref_texts(ref_path)
    print(f"Loaded {len(ref_texts)} reference texts from {ref_path}")

    normalizer = EnglishTextNormalizer()

    # 2. Load model.
    try:
        asr_model = load_nemo_model(args.model_path)
    except ImportError:
        print("Please run: pip install nemo_toolkit['all']")
        return

    # Disable the RNNT greedy CUDA Graph decoder and rebuild the decoding component.
    try:
        decoding_cfg = deepcopy(asr_model.cfg.decoding)
        with open_dict(decoding_cfg):
            if hasattr(decoding_cfg, "strategy"):
                decoding_cfg.strategy = "greedy"
            elif isinstance(decoding_cfg, dict) and "strategy" in decoding_cfg:
                decoding_cfg["strategy"] = "greedy"

            if hasattr(decoding_cfg, "greedy") and hasattr(decoding_cfg.greedy, "use_cuda_graph_decoder"):
                decoding_cfg.greedy.use_cuda_graph_decoder = False

            if hasattr(decoding_cfg, "decoding") and hasattr(decoding_cfg.decoding, "use_cuda_graph_decoder"):
                decoding_cfg.decoding.use_cuda_graph_decoder = False

        if hasattr(asr_model, 'change_decoding_strategy'):
            asr_model.change_decoding_strategy(decoding_cfg=decoding_cfg)
    except Exception as _e:
        # Do not block the main flow; some CTC models or older versions may not expose these fields.
        pass

    # 3. Scan files.
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        print("No wav files found.")
        return

    # 4. Resume check.
    processed_files = set()
    write_header = True

    csv_file_handle = open(detailed_log_path, "w", encoding="utf-8-sig", newline="")
    writer = csv.writer(csv_file_handle)
    if write_header:
        writer.writerow(["Filename", "WER", "S", "D", "I", "H", "Ref_Norm", "Hyp_Norm"])
        csv_file_handle.flush()

    files_to_process = [f for f in wav_files if os.path.basename(f) not in processed_files]
    print(f"Remaining files to process: {len(files_to_process)}")

    # 5. Sort by real duration in descending order and cache durations.
    print("Sorting files by duration (longest first)...")
    dur_map = {}
    for f in tqdm(files_to_process, desc="Reading headers"):
        dur_map[f] = get_audio_duration(f)
    files_to_process.sort(key=lambda f: dur_map[f], reverse=True)

    # 6. Dynamic batch loop.
    batch_paths = []
    batch_refs = []

    # Current batch size limit, determined by the first audio in the batch.
    current_batch_limit = 16

    predictions: List[str] = []
    references: List[str] = []
    evaluated_ids: List[str] = []

    def flush_batch():
        nonlocal batch_paths, batch_refs
        if not batch_paths: return

        # Inference.
        try:
            # batch_size only tells NeMo how to split the batch.
            # We already control batch_paths manually, so pass its length directly.
            hypotheses = asr_model.transcribe(audio=batch_paths, batch_size=len(batch_paths))
            if isinstance(hypotheses, tuple): hypotheses = hypotheses[0]
        except Exception as e:
            print(f"\n[ERROR] Batch inference failed: {e}")
            raise e

        # Write results.
        for i, hyp_text in enumerate(hypotheses):
            wav_path = batch_paths[i]
            ref_text = batch_refs[i]
            p_norm = normalizer(normalize_text(hyp_text))
            r_norm = normalizer(normalize_text(ref_text))

            predictions.append(p_norm)
            references.append(r_norm)
            evaluated_ids.append(os.path.splitext(os.path.basename(wav_path))[0])

            if not r_norm.strip():
                s, d, i_cnt, h_cnt, wer_val = 0, 0, 0, 0, 0.0
            else:
                out = process_words(r_norm, p_norm)
                s, d, i_cnt, h_cnt = out.substitutions, out.deletions, out.insertions, out.hits
                wer_val = out.wer

            writer.writerow([
                os.path.basename(wav_path), f"{wer_val:.4f}", s, d, i_cnt, h_cnt, r_norm, p_norm
            ])

            if os.path.basename(wav_path) in PRINT_TARGETS:
                write_log(f"[DEMO] {os.path.basename(wav_path)} WER: {wer_val:.2%}")

        csv_file_handle.flush()
        batch_paths = []
        batch_refs = []

    print(f"Start Processing with Dynamic Batching (Capacity Factor: {args.capacity})...")

    for wav_path in tqdm(files_to_process, desc="Evaluating"):
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        gt_text = ref_texts.get(utt_id)
        if gt_text is None:
            write_log(f"[WARN] Missing reference text for {utt_id}, skipped.")
            continue

        # --- Dynamic batch core logic ---
        if len(batch_paths) == 0:
            # The first file of a new batch is the longest one because files are sorted descending.
            # Use its duration to decide how many files this batch can hold.
            duration = dur_map.get(wav_path, get_audio_duration(wav_path))

            # Compute limit = capacity / duration.
            # Example: 140 / 35s = 4.
            # Example: 140 / 7s = 20.
            calculated_limit = int(args.capacity / duration)

            # Clamp to [1, MAX_LIMIT].
            current_batch_limit = max(1, min(calculated_limit, MAX_BATCH_SIZE_LIMIT))

        # Add the current file.
        batch_paths.append(wav_path)
        batch_refs.append(gt_text)

        # Flush when the batch is full.
        if len(batch_paths) >= current_batch_limit:
            flush_batch()

    flush_batch() # Flush the remaining files.

    csv_file_handle.close()
    print(f"Done. Results in {detailed_log_path}")

    with open(hyp_path, 'w', encoding='utf-8') as f:
        for utt_id, line in zip(evaluated_ids, predictions):
            f.write(f"{line} ({utt_id})\n")
    print(f"{len(predictions)} predcition texts have been written to {hyp_path}")

    if not predictions:
        write_log("No evaluated samples. Please check wav filenames and ref_path utt_ids.")
        return

    final_wer = wer(references, predictions)

    write_log("\n" + "=" * 60)
    write_log(f"Recon wav dir: {h.test_wav_output_dir}")
    write_log(f"Total Evaluated Samples: {len(predictions)}")
    write_log(f"Final WER: {final_wer:.4%}")
    write_log("=" * 60)

if __name__ == "__main__":
    main()
