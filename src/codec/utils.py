import os
import json
import random
import yaml
import math
import numpy as np
import torch
import shutil
from src.utils.analyze_codebook import CodebookStatsAccumulator

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_hparams(config_path: str):
    """Load hyperparameters from a YAML (preferred) or JSON config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith((".yaml", ".yml")):
            data = yaml.safe_load(f)
        elif config_path.endswith(".json"):
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file type: {config_path}")
    if data is None:
        raise ValueError(f"Empty config file: {config_path}")
    return AttrDict(data)

def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def get_state_dict(model):
    if hasattr(model, 'module'):
        return model.module.state_dict()
    return model.state_dict()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_requires_grad(m, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)

def unwrap_ddp(m):
    return m.module if hasattr(m, "module") else m

def set_quantizer_mode(quantizer, stochastic: bool, temperature: float):
    quantizer_core = unwrap_ddp(quantizer)
    if hasattr(quantizer_core, "set_stochastic_mode"):
        quantizer_core.set_stochastic_mode(stochastic=stochastic, temperature=temperature)

def get_codebook_sizes(quantizer) -> tuple[int, ...]:
    codebook_sizes = getattr(quantizer, "codebook_sizes", ())
    if callable(codebook_sizes):
        codebook_sizes = codebook_sizes()
    return tuple(int(x) for x in codebook_sizes)

def build_stats_accumulator(codebook_sizes: tuple[int, ...]):
    if not codebook_sizes:
        return None
    return CodebookStatsAccumulator(
        codebook_size=codebook_sizes[0],
        num_quantizers=len(codebook_sizes),
        quantizer_dim=1 if len(codebook_sizes) > 1 else None,
    )

def compute_file_bits(codes: torch.Tensor | None, codebook_sizes: tuple[int, ...]) -> int:
    if codes is None or not codebook_sizes:
        return 0
    bits_per_frame = sum(math.ceil(math.log2(size)) for size in codebook_sizes)
    if codes.dim() == 2:
        num_frames = int(codes.shape[1])
    elif codes.dim() == 3:
        num_frames = int(codes.shape[-1])
    else:
        raise RuntimeError(f"Unexpected codes shape: {tuple(codes.shape)}")
    return bits_per_frame * num_frames

def infer_codec_vocos_state_paths(resume_path: str):
    d = os.path.dirname(resume_path)
    base = os.path.basename(resume_path)

    if base.startswith("state_"):
        state_path = resume_path
        codec_path = os.path.join(d, "codec_" + base[len("state_"):])
        vocos_path = os.path.join(d, "vocos_" + base[len("state_"):])
    elif base.startswith("codec_"):
        codec_path = resume_path
        vocos_path = os.path.join(d, "vocos_" + base[len("codec_"):])
        state_path = os.path.join(d, "state_" + base[len("codec_"):])
    elif base.startswith("vocos_"):
        vocos_path = resume_path
        codec_path = os.path.join(d, "codec_" + base[len("vocos_"):])
        state_path = os.path.join(d, "state_" + base[len("vocos_"):])
    else:
        raise ValueError(f"--resume_from_checkpoint must point to codec_* or vocos_* or state_*, got: {resume_path}")

    return codec_path, vocos_path, state_path

def infer_codec_state_paths(resume_path: str):
    """
    Given a codec_* or state_* path, return (codec_path, state_path).
    """
    base = os.path.basename(resume_path)
    d = os.path.dirname(resume_path)

    if base.startswith("codec_"):
        codec_path = resume_path
        state_path = os.path.join(d, "state_" + base[len("codec_"):])
    elif base.startswith("state_"):
        state_path = resume_path
        codec_path = os.path.join(d, "codec_" + base[len("state_"):])
    else:
        raise ValueError(f"resume_from_checkpoint must point to codec_* or state_*, got: {resume_path}")

    return codec_path, state_path
    
