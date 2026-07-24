import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import itertools
import math
import os
import socket
import sys
import time

import jiwer
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.codec.asr_model import Parakeet
from src.codec.quantizer import build_quantizer
from src.codec.streaming_decoder import StreamingDecoder
from src.codec.streaming_encoder import StreamingEncoder
from src.codec.utils import (
    build_env, get_state_dict, load_checkpoint, load_hparams, save_checkpoint,
    set_seed, set_requires_grad, unwrap_ddp,
    set_quantizer_mode as _set_quantizer_mode,
    infer_codec_vocos_state_paths as _infer_codec_state_paths,
)
from src.data.dataset import Dataset, get_dataset_filelist
from src.data.web_dataset import make_wds_audio_dataset
from src.vocos.streaming_features import CausalMelSpectrogramFeatures, CausalSingleMelLoss, pad_mel_to_multiple
from src.vocos.streaming_heads import StreamingISTFTHead
from src.vocos.streaming_models import StreamingVocosBackbone

torch.backends.cudnn.benchmark = True


class Reward_Model:
    def __init__(self, model):
        self.model = model
        self.total_count = 0
        self.empty_ref_count = 0
        self.empty_hyp_count = 0

    def reset_stats(self):
        self.total_count = 0
        self.empty_ref_count = 0
        self.empty_hyp_count = 0

    def get_and_reset_counts(self):
        counts = (self.total_count, self.empty_ref_count, self.empty_hyp_count)
        self.reset_stats()
        return counts

    def normalize(self, x, eps=1e-7):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + eps)

    def compute_reward(self, ref_wavs, hyp_wavs):
        if ref_wavs.ndim == 3:
            ref_wavs = ref_wavs.squeeze(1)
        if hyp_wavs.ndim == 3:
            hyp_wavs = hyp_wavs.squeeze(1)

        hyp_transcription = self.model.transcribe(hyp_wavs)
        ref_transcription = self.model.transcribe(ref_wavs)

        rewards = []
        for r, h in zip(ref_transcription, hyp_transcription):
            self.total_count += 1
            r_clean = r.strip()
            h_clean = h.strip()

            if len(r_clean) == 0:
                self.empty_ref_count += 1
            if len(h_clean) == 0:
                self.empty_hyp_count += 1

            if len(r_clean) == 0:
                if len(h_clean) == 0:
                    rewards.append(0.0)
                else:
                    rewards.append(-1.0)
                continue

            if len(h_clean) == 0:
                rewards.append(-1.0)
                continue

            try:
                wer = jiwer.wer(r_clean, h_clean)
                rewards.append(-wer)
            except Exception:
                rewards.append(0.0)

        return torch.tensor(rewards, device=ref_wavs.device)


def train(rank, local_rank, world_size, h, resume_from_checkpoint: str = ""):
    
    rl_h = h.rl
    device = torch.device("cuda", local_rank)

    set_seed(h.seed)

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

    mel_spectrogram = CausalMelSpectrogramFeatures(
        sample_rate=h.sampling_rate,
        n_fft=h.n_fft,
        win_length=h.win_size,
        hop_length=h.hop_size,
        n_mels=h.num_mels,
    ).to(device)

    asr_model = Parakeet(device=device)
    reward_model = Reward_Model(asr_model)

    steps = 0
    set_requires_grad(encoder, True)
    set_requires_grad(quantizer, False)
    set_requires_grad(decoder, False)
    set_requires_grad(vocosbackbone, False)
    set_requires_grad(istfthead, False)

    if not hasattr(quantizer, "rfsq"):
        raise ValueError("RL training expects an RFSQ quantizer with rfsq.layers.")

    for layer in quantizer.rfsq.layers:
        set_requires_grad(layer.project_in, True)
        set_requires_grad(layer.project_out, False)

    trainable_params = [p for p in itertools.chain(encoder.parameters(), quantizer.parameters()) if p.requires_grad]
    if rank == 0:
        train_params = sum(p.numel() for p in trainable_params)

    learning_rate = float(rl_h["learning_rate"])
    optim_g = torch.optim.AdamW(trainable_params, learning_rate, betas=[h.adam_b1, h.adam_b2])

    mel_loss = CausalSingleMelLoss(
        sample_rate=h.sampling_rate,
        n_fft=h.n_fft,
        win_length=h.win_size,
        hop_length=h.hop_size,
        n_mels=h.num_mels,
    ).to(device)

    total_steps = int(rl_h["max_training_steps"])
    scheduler_g = torch.optim.lr_scheduler.OneCycleLR(
        optim_g,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=h.pct_start,
        div_factor=h.div_factor,
        final_div_factor=h.final_div_factor,
        anneal_strategy="cos",
        last_epoch=-1,
    )

    if resume_from_checkpoint:
        print("resume from checkpoint")
        cp_codec, cp_vocos, cp_state = _infer_codec_state_paths(resume_from_checkpoint)

        if not os.path.isfile(cp_codec):
            raise FileNotFoundError(f"codec checkpoint not found: {cp_codec}")
        if not os.path.isfile(cp_vocos):
            raise FileNotFoundError(f"vocos checkpoint not found: {cp_vocos}")
        if not os.path.isfile(cp_state):
            raise FileNotFoundError(f"state checkpoint not found: {cp_state}")

        state_dict_codec = load_checkpoint(cp_codec, device)
        state_dict_vocos = load_checkpoint(cp_vocos, device)
        state_dict_state = load_checkpoint(cp_state, device)

        encoder.load_state_dict(state_dict_codec["encoder"], strict=True)
        quantizer.load_state_dict(state_dict_codec["quantizer"], strict=True)
        decoder.load_state_dict(state_dict_codec["decoder"], strict=True)
        vocosbackbone.load_state_dict(state_dict_vocos["vocosbackbone"], strict=True)
        istfthead.load_state_dict(state_dict_vocos["istfthead"], strict=True)
        steps = int(state_dict_state["steps"])
        optim_g.load_state_dict(state_dict_state["optim_g"])
        scheduler_g.load_state_dict(state_dict_state["scheduler_g"])
    else:
        cp_codec = load_checkpoint(rl_h["checkpoint_path_codec"], device)
        cp_vocos = load_checkpoint(rl_h["checkpoint_path_vocos"], device)

        encoder.load_state_dict(cp_codec["encoder"], strict=True)
        quantizer.load_state_dict(cp_codec["quantizer"], strict=True)
        decoder.load_state_dict(cp_codec["decoder"], strict=True)
        vocosbackbone.load_state_dict(cp_vocos["vocosbackbone"], strict=True)
        istfthead.load_state_dict(cp_vocos["istfthead"], strict=True)

    encoder = DDP(encoder, device_ids=[local_rank], find_unused_parameters=h.ddp_find_unused_parameters) if world_size > 1 and dist.is_initialized() else encoder
    quantizer = DDP(quantizer, device_ids=[local_rank], find_unused_parameters=h.ddp_find_unused_parameters) if world_size > 1 and dist.is_initialized() else quantizer

    validation_filelist = get_dataset_filelist(h.input_validation_wav_list)

    trainset = make_wds_audio_dataset(
        h.train_shards_path,
        h.sampling_rate,
        rl_h["train_segment_size"],
        h.sample_shuffle,
        True,
        True,
        h.seed,
    )

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        batch_size=rl_h["train_batch_size"],
        pin_memory=True,
        drop_last=False,
        persistent_workers=(h.num_workers > 0),
        prefetch_factor=2 if h.num_workers > 0 else None,
    )

    validset = Dataset(
        validation_filelist,
        rl_h["valid_segment_size"],
        h.sampling_rate,
        h.hop_size,
        math.prod(h.down_ratio),
        train=False,
        shuffle=False,
        n_cache_reuse=0,
        device=device,
    )
    if world_size > 1 and dist.is_initialized():
        valid_sampler = DistributedSampler(validset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        valid_sampler = None

    validation_loader = DataLoader(
        validset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=valid_sampler,
        batch_size=rl_h["valid_batch_size"],
        pin_memory=True,
        drop_last=False,
    )

    sw = None
    if rank == 0:
        sw = SummaryWriter(h.logs_dir)

    decoder.eval()
    vocosbackbone.eval()
    istfthead.eval()

    group_size = int(rl_h["group_size"])
    validation_interval = int(rl_h["validation_interval"])
    temperature = float(rl_h["temperature"])
    rl_weight = float(rl_h["rl_weight"])
    rl_mel_l1_weight = float(rl_h["mel_l1_weight"])
    use_gradient_clipping = bool(rl_h["use_gradient_clipping"])
    gradient_clip_norm = float(rl_h["gradient_clip_norm"])

    if rank == 0:
        print(f"GRPO Training Initialized. Group Size: {group_size}")

    train_iter = iter(train_loader)
    while steps < total_steps:
        _set_quantizer_mode(quantizer, stochastic=True, temperature=temperature)

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        encoder.train()
        quantizer.train()
        start_b = time.time()

        y = batch
        y = torch.autograd.Variable(y.to(device, non_blocking=True))
        y = y.squeeze()

        y_repeated = y.repeat_interleave(group_size, dim=0)
        mel_repeated = mel_spectrogram(y_repeated)
        mel_repeated, _ = pad_mel_to_multiple(mel_repeated, math.prod(h.down_ratio))

        latent = encoder(mel_repeated)
        quantizer_out = quantizer(latent)

        if quantizer_out.codes is None:
            raise RuntimeError("RL training requires sampled quantizer codes.")
        frames = quantizer_out.codes.size(-1)

        saved_log_probs = unwrap_ddp(quantizer).rfsq.saved_log_probs
        if not saved_log_probs:
            raise RuntimeError("RFSQ saved_log_probs is empty. Make sure stochastic mode is enabled.")
        log_probs = 0.0
        for layer_log_probs in saved_log_probs:
            log_probs = log_probs + layer_log_probs.sum(dim=tuple(range(1, layer_log_probs.ndim)))
        log_probs = log_probs / frames

        mel_latent = decoder(quantizer_out.z_q)
        vocoder_emb = vocosbackbone(mel_latent)
        y_g = istfthead(vocoder_emb)
        length = min(y_g.shape[-1], y_repeated.shape[-1])
        y_g = y_g[..., :length]
        y_repeated = y_repeated[..., :length]
        if rl_h["tail_crop_size"] > 0:
            y_g = y_g[..., :-rl_h["tail_crop_size"]]
            y_repeated = y_repeated[..., :-rl_h["tail_crop_size"]]

        with torch.no_grad():
            rewards = reward_model.compute_reward(y_repeated, y_g)

        rewards_view = rewards.view(-1, group_size)
        mean_rewards = rewards_view.mean(dim=1, keepdim=True)
        std_rewards = rewards_view.std(dim=1, keepdim=True, unbiased=False) + 1e-4
        advantages = (rewards_view - mean_rewards) / std_rewards
        advantages = advantages.view(-1)

        pg_loss = -(advantages.detach() * log_probs).mean()
        recon_loss = rl_mel_l1_weight * mel_loss(y_repeated, y_g)
        loss = rl_weight * pg_loss + recon_loss

        optim_g.zero_grad()
        loss.backward()
        if use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=gradient_clip_norm)
        optim_g.step()
        scheduler_g.step()

        if rank == 0:
            if steps % h.summary_interval == 0:
                with torch.no_grad():
                    total_error = loss.item()
                    mel_l1_error = recon_loss.item()
                    pg_error = pg_loss.item()
                    avg_reward = rewards.mean().item()

                print(
                    "Steps : {:d}, Total Loss : {:4.3f}, Mel L1 : {:4.3f}, GRPO PG Loss : {:4.3f}, "
                    "Avg Reward : {:4.3f}, s/b : {:4.3f}".format(
                        steps, total_error, mel_l1_error, pg_error, avg_reward, time.time() - start_b
                    )
                )

            if steps % h.summary_interval == 0:
                sw.add_scalar("Training/Total_Loss", loss.item(), steps)
                sw.add_scalar("Training/Reconstruction_Loss", recon_loss.item(), steps)
                sw.add_scalar("Training/GRPO_PG_Loss", pg_loss.item(), steps)
                sw.add_scalar("Training/Average_Reward", rewards.mean().item(), steps)
                sw.add_scalar("train/lr", scheduler_g.get_last_lr()[0], steps)

        steps += 1

        if steps % validation_interval == 0 and steps != 0:
            encoder.eval()
            quantizer.eval()
            _set_quantizer_mode(quantizer, stochastic=False, temperature=temperature)
            torch.cuda.empty_cache()
            reward_model.reset_stats()

            val_mel_l1_err_tot = 0
            val_wer_tot = 0
            val_num = 0

            with torch.no_grad():
                for batch in validation_loader:
                    y = batch.to(device)
                    mel = mel_spectrogram(y)
                    mel, _ = pad_mel_to_multiple(mel, math.prod(h.down_ratio))
                    latent = encoder(mel)
                    quantizer_out = quantizer(latent)
                    mel_latent = decoder(quantizer_out.z_q)
                    vocoder_emb = vocosbackbone(mel_latent)
                    y_g = istfthead(vocoder_emb)
                    length = min(y_g.shape[-1], y.shape[-1])
                    y_g = y_g[..., :length]
                    y = y[..., :length]

                    if rl_h["tail_crop_size"] > 0:
                        y_g = y_g[..., :-rl_h["tail_crop_size"]]
                        y = y[..., :-rl_h["tail_crop_size"]]

                    bsz = y.size(0)
                    val_num += bsz
                    val_mel_l1_err_tot += mel_loss(y, y_g).item() * bsz
                    val_wer_tot += -(reward_model.compute_reward(y, y_g)).sum().item()

            stats_total, empty_ref_count, empty_hyp_count = reward_model.get_and_reset_counts()

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                t = torch.tensor(
                    [
                        val_mel_l1_err_tot,
                        val_wer_tot,
                        float(val_num),
                        float(stats_total),
                        float(empty_ref_count),
                        float(empty_hyp_count),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                (
                    val_mel_l1_err_tot,
                    val_wer_tot,
                    val_num,
                    stats_total,
                    empty_ref_count,
                    empty_hyp_count,
                ) = t.tolist()

            if rank == 0:
                val_mel_l1_err = val_mel_l1_err_tot / val_num
                val_wer = val_wer_tot / val_num

                sw.add_scalar("Validation/Codec_Mel_loss", val_mel_l1_err, steps)
                sw.add_scalar("Validation/WER", val_wer, steps)

                if stats_total > 0:
                    empty_ref_rate = empty_ref_count / stats_total
                    empty_hyp_rate = empty_hyp_count / stats_total
                else:
                    empty_ref_rate = 0.0
                    empty_hyp_rate = 0.0
                print(f"  >> Empty Reference Rate (Data Quality): {empty_ref_rate * 100:.2f}%")
                print(f"  >> Empty Hypothesis Rate (Model Silence): {empty_hyp_rate * 100:.2f}%")
                sw.add_scalar("Stats/Empty_Reference_Rate", empty_ref_rate, steps)
                sw.add_scalar("Stats/Empty_Hypothesis_Rate", empty_hyp_rate, steps)

                save_checkpoint(
                    f"{h.checkpoint_path}/codec_last",
                    {
                        "encoder": get_state_dict(encoder),
                        "quantizer": get_state_dict(quantizer),
                        "decoder": get_state_dict(decoder),
                    },
                )
                save_checkpoint(
                    f"{h.checkpoint_path}/vocos_last",
                    {"vocosbackbone": get_state_dict(vocosbackbone), "istfthead": get_state_dict(istfthead)},
                )
                save_checkpoint(
                    f"{h.checkpoint_path}/state_last",
                    {
                        "optim_g": optim_g.state_dict(),
                        "scheduler_g": scheduler_g.state_dict(),
                        "steps": steps,
                    },
                )
            torch.cuda.empty_cache()


def main():
    print("Initializing Training Process..")
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="", help="Path to config YAML")
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="Path to codec_* or vocos_* or state_* checkpoint")
    args = parser.parse_args()
    config_file = args.config
    h = load_hparams(config_file)

    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ

    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        print("=" * 80)
        print(f"[Distributed Setup] Hostname: {socket.gethostname()}")
        print(f"[Distributed Setup] RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
        print(f"[Distributed Setup] MASTER_ADDR={os.environ.get('MASTER_ADDR', 'NOT SET')}")
        print(f"[Distributed Setup] MASTER_PORT={os.environ.get('MASTER_PORT', 'NOT SET')}")
        print("=" * 80)

        backend = h.ddp_backend
        dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)
    else:
        rank, world_size, local_rank = 0, 1, 0

    exp_name = h.exp_name
    runs_root = h.runs_root
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_id = h.run_id or f"{timestamp}__seed{h.seed}__gpu{world_size}"
    run_dir = os.path.join(runs_root, exp_name, run_id)

    if is_distributed:
        obj_list = [run_dir if rank == 0 else None]
        dist.broadcast_object_list(obj_list, src=0)
        run_dir = obj_list[0]

    meta_dir = os.path.join(run_dir, "meta")
    logs_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    samples_dir = os.path.join(run_dir, "samples")
    metrics_dir = os.path.join(run_dir, "metrics")

    if rank == 0:
        os.makedirs(meta_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        try:
            build_env(config_file, "config.yaml", meta_dir)
        except Exception:
            import shutil

            shutil.copyfile(config_file, os.path.join(meta_dir, "config.yaml"))

        with open(os.path.join(meta_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(h), f, allow_unicode=True, sort_keys=False)

        with open(os.path.join(meta_dir, "env.txt"), "w", encoding="utf-8") as f:
            f.write(f"torch={torch.__version__}\n")
            f.write(f"cuda_available={torch.cuda.is_available()}\n")
            f.write(f"cuda_version={torch.version.cuda}\n")
            f.write(f"cudnn_version={torch.backends.cudnn.version()}\n")
            try:
                f.write(f"device_count={torch.cuda.device_count()}\n")
            except Exception:
                pass

    if is_distributed:
        dist.barrier()

    h.run_dir = run_dir
    h.meta_dir = meta_dir
    h.logs_dir = logs_dir
    h.samples_dir = samples_dir
    h.metrics_dir = metrics_dir
    h.checkpoint_path = ckpt_dir

    train(rank, local_rank, world_size, h, args.resume_from_checkpoint)

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
