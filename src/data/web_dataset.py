"""WebDataset pipeline for sharded audio stored locally or on remote storage.

Remote shards can be addressed through an rclone remote and streamed through
WebDataset without mounting or staging them on local disk.
"""

import re
import random
import hashlib
from typing import List, Dict, Any

import torch
import torchaudio
import webdataset as wds
import logging
import torch.distributed as dist

TARGET_SR = 16000
SEGMENT_SIZE = 16000 * 3
SAMPLE_SHUFFLE = 5000
SHARD_SHUFFLE = True
REMOTE_AUTO_PIPE = True

RCLONE_RETRIES = 5
RCLONE_LOW_LEVEL_RETRIES = 20
RCLONE_TIMEOUT = "20s"
RCLONE_CONTIMEOUT = "60s"

def load_shard_list(list_file: str) -> List[str]:
    urls = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            urls.append(s)
    if not urls:
        raise RuntimeError(f"No shards found in: {list_file}")
    return urls


def maybe_pipe_rclone(url: str) -> str:
    """
    Convert an rclone path such as remote_name:path/to/train-000000.tar to a
    WebDataset pipe URL for direct remote streaming.
    """
    if url.startswith("pipe:"):
        return url
    if re.match(r"^[A-Za-z0-9._-]+:.*\.tar$", url):
        return (
            f"pipe:rclone cat {url} "
            f"--retries {RCLONE_RETRIES} " 
            f"--low-level-retries {RCLONE_LOW_LEVEL_RETRIES} "
            f"--timeout {RCLONE_TIMEOUT} "
            f"--contimeout {RCLONE_CONTIMEOUT}"
        )
    return url


class ResamplerCache:
    """Cache Resample modules to avoid rebuilding them for every sample."""
    def __init__(self, target_sr: int):
        self.target_sr = target_sr
        self._cache: Dict[int, torchaudio.transforms.Resample] = {}

    def __call__(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.target_sr:
            return wav
        if sr not in self._cache:
            self._cache[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
        return self._cache[sr](wav)


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    # wav: [C, T] or [T]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def random_crop_or_pad(wav: torch.Tensor, segment_size: int, rng: random.Random) -> torch.Tensor:
    assert wav.dim() == 2 and wav.size(0) == 1, f"expect [1,T], got {tuple(wav.shape)}"
    T = wav.size(1)

    if segment_size <= 0:
        raise ValueError(f"segment_size must be > 0, got {segment_size}")

    # Edge case: empty audio should not happen, but return zeros as a fallback.
    if T == 0:
        return wav.new_zeros((1, segment_size))

    if T == segment_size:
        return wav

    if T > segment_size:
        start = rng.randint(0, T - segment_size)
        return wav[:, start:start + segment_size]

    # T < segment_size: tile the waveform.
    # Number of repeats needed.
    reps = (segment_size + T - 1) // T  # ceil(segment_size / T)
    out = wav.repeat(1, reps)[:, :segment_size]
    return out

def sanitize_keys(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Some tar files decode with keys such as 'filename.flac' instead of 'flac'.
    Rename .flac/.wav keys to 'flac'/'wav' so .decode(wds.torch_audio) can recognize them.
    """
    for key in list(sample.keys()):
        # Already normalized.
        if key in ['flac', 'wav']:
            continue
        
        # If the key is xxx.flac, copy it to sample['flac'].
        if key.endswith('.flac'):
            sample['flac'] = sample[key]
        elif key.endswith('.wav'):
            sample['wav'] = sample[key]
            
    return sample

def get_wav_only(sample: Dict[str, Any]) -> torch.Tensor:
    """Return the wav tensor directly without tuple wrapping."""
    return sample["wav"]

def log_and_continue(exn):
    """Log the exception and let WebDataset continue to the next sample/shard."""
    logging.warning(f"Handling WebDataset error: {repr(exn)}. Skipping sample/shard.")
    return True  # True means ignore the error and continue.


def _stable_int_hash(*parts: object) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "big")

# =========================
# Core sample mapping function
# =========================

def make_audio_segment_mapper(
    target_sr: int,
    segment_size: int,
    seed: int,
    cycle_id: int,
    rank: int,
    worker_id: int,
):
    """
    Return a closure that maps an input sample dict to an output sample dict with a wav field.
    To keep multi-worker sampling reproducible but not identical, the seed includes worker context.
    """
    resample = ResamplerCache(target_sr)

    def _map(sample: Dict[str, Any]) -> Dict[str, Any]:

        sample_seed = _stable_int_hash(
            seed,
            cycle_id,
            rank,
            worker_id,
            sample.get("__url__", ""),
            sample.get("__key__", ""),
        ) % (2**31)
        rng = random.Random(sample_seed)

        audio = sample["audio"]  # (wav, sr) after decode
        wav, sr = audio  # wav: [C,T]
        if not isinstance(wav, torch.Tensor):
            wav = torch.as_tensor(wav)

        wav = wav.to(torch.float32)
        wav = to_mono(wav)
        wav = resample(wav, int(sr))
        wav = random_crop_or_pad(wav, segment_size, rng)

        # Keep only the waveform; sample rate is no longer needed.
        sample["wav"] = wav  # [1, segment_size]
        return sample

    return _map


class WDSAudioDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        urls: List[str],
        target_sr: int,
        segment_size: int,
        sample_shuffle: int,
        shard_shuffle: bool,
        seed: int,
    ):
        super().__init__()
        self.urls = urls
        self.target_sr = target_sr
        self.segment_size = segment_size
        self.sample_shuffle = sample_shuffle
        self.shard_shuffle = shard_shuffle
        self.seed = seed

    def __iter__(self):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        cycle_id = 0
        while True:
            cycle_urls = list(self.urls)
            if self.shard_shuffle:
                random.Random(self.seed + cycle_id).shuffle(cycle_urls)

            cycle_urls = cycle_urls[rank::world_size]
            cycle_urls = cycle_urls[worker_id::num_workers]

            if not cycle_urls:
                raise RuntimeError("No shards assigned to the current rank/worker.")

            mapper = make_audio_segment_mapper(
                self.target_sr,
                self.segment_size,
                self.seed,
                cycle_id,
                rank,
                worker_id,
            )

            ds = wds.DataPipeline(
                wds.SimpleShardList(cycle_urls),
                wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(self.sample_shuffle),
                wds.map(sanitize_keys, handler=log_and_continue),
                wds.decode(wds.torch_audio, handler=log_and_continue),
                wds.rename(audio="flac;wav", handler=log_and_continue),
                wds.map(mapper, handler=log_and_continue),
                wds.map(get_wav_only, handler=log_and_continue),
            )

            yield from iter(ds)
            cycle_id += 1


def make_wds_audio_dataset(
    shards_list_file: str,
    target_sr: int = TARGET_SR,
    segment_size: int = SEGMENT_SIZE,
    sample_shuffle: int = SAMPLE_SHUFFLE,
    shard_shuffle: bool = SHARD_SHUFFLE,
    remote_auto_pipe: bool = REMOTE_AUTO_PIPE,
    seed: int = 1234,
):
    urls = load_shard_list(shards_list_file)
    if remote_auto_pipe:
        urls = [maybe_pipe_rclone(u) for u in urls]

    ds = WDSAudioDataset(
        urls=urls,
        target_sr=target_sr,
        segment_size=segment_size,
        sample_shuffle=sample_shuffle,
        shard_shuffle=shard_shuffle,
        seed=seed,
    )

    return ds
