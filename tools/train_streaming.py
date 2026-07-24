import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import math
import itertools
import os
import time
import argparse
import yaml
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.dataset import Dataset, get_dataset_filelist
from src.data.web_dataset import make_wds_audio_dataset
from src.codec.streaming_encoder import StreamingEncoder
from src.codec.streaming_decoder import StreamingDecoder
from src.codec.quantizer import build_quantizer
from src.codec.discriminators import MultiScaleDiscriminator, MultiPeriodDiscriminator, MultiResolutionDiscriminator
from src.codec.utils import (
    load_hparams, build_env, load_checkpoint, save_checkpoint, get_state_dict,
    set_seed, set_requires_grad, set_quantizer_mode as _set_quantizer_mode,
    infer_codec_vocos_state_paths as _infer_codec_state_paths,
)
from src.vocos.streaming_features import CausalMelSpectrogramFeatures, CausalSingleMelLoss, pad_mel_to_multiple
from src.vocos.streaming_heads import StreamingISTFTHead
from src.vocos.streaming_models import StreamingVocosBackbone
from src.loss.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss
import socket

torch.backends.cudnn.benchmark = True


def train(rank, local_rank, world_size, h, resume_from_checkpoint: str = ""):
    
    device = torch.device("cuda", local_rank)

    set_seed(h.seed)

    encoder = StreamingEncoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    decoder = StreamingDecoder(h).to(device)

    vocosbackbone = StreamingVocosBackbone(
        input_channels=h.vocos_backbone_input_channels,
        dim=h.vocos_backbone_dim,
        intermediate_dim=h.vocos_backbone_intermediate_dim,
        num_layers=h.vocos_backbone_num_layers
    ).to(device)

    istfthead = StreamingISTFTHead(
        dim=h.vocos_head_dim,
        n_fft=h.vocos_head_n_fft,
        win_length=h.vocos_head_win_length,
        hop_length=h.vocos_head_hop_length
    ).to(device)

    mel_spectrogram = CausalMelSpectrogramFeatures(sample_rate=h.sampling_rate, n_fft=h.n_fft, win_length=h.win_size, hop_length=h.hop_size, n_mels=h.num_mels).to(device)
    mel_loss = CausalSingleMelLoss(sample_rate=h.sampling_rate, n_fft=h.n_fft, win_length=h.win_size, hop_length=h.hop_size, n_mels=h.num_mels).to(device)

    msd = MultiScaleDiscriminator(h).to(device)
    mpd = MultiPeriodDiscriminator(h).to(device)
    mrd = MultiResolutionDiscriminator(h).to(device)

    discriminator_loss = DiscriminatorLoss()
    feature_loss = FeatureMatchingLoss()
    generator_loss = GeneratorLoss()

    if rank == 0:
        os.makedirs(h.checkpoint_path, exist_ok=True)
        print("codec checkpoints directory : ", h.checkpoint_path)

    steps = 0

    cp_codec, cp_vocos, cp_state = None, None, None

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

        msd.load_state_dict(state_dict_state["msd"], strict=True)
        mpd.load_state_dict(state_dict_state["mpd"], strict=True)
        mrd.load_state_dict(state_dict_state["mrd"], strict=True)

        steps = int(state_dict_state["steps"])

    encoder = DDP(encoder, device_ids=[local_rank], find_unused_parameters=h.ddp_find_unused_parameters) if world_size > 1 and dist.is_initialized() else encoder
    quantizer = DDP(quantizer, device_ids=[local_rank], find_unused_parameters=h.ddp_find_unused_parameters) if world_size > 1 and dist.is_initialized() else quantizer
    decoder = DDP(decoder, device_ids=[local_rank], find_unused_parameters=h.ddp_find_unused_parameters) if world_size > 1 and dist.is_initialized() else decoder

    vocosbackbone = DDP(vocosbackbone, device_ids=[local_rank], find_unused_parameters=h.ddp_find_unused_parameters) if world_size > 1 and dist.is_initialized() else vocosbackbone
    istfthead = DDP(istfthead, device_ids=[local_rank], find_unused_parameters=h.ddp_find_unused_parameters) if world_size > 1 and dist.is_initialized() else istfthead

    msd = DDP(msd, device_ids=[local_rank], find_unused_parameters=h.ddp_find_unused_parameters) if world_size > 1 and dist.is_initialized() else msd
    mpd = DDP(mpd, device_ids=[local_rank], find_unused_parameters=h.ddp_find_unused_parameters) if world_size > 1 and dist.is_initialized() else mpd
    mrd = DDP(mrd, device_ids=[local_rank], find_unused_parameters=h.ddp_find_unused_parameters) if world_size > 1 and dist.is_initialized() else mrd

    optim_g = torch.optim.AdamW(
        itertools.chain(encoder.parameters(), quantizer.parameters(), decoder.parameters(), vocosbackbone.parameters(), istfthead.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters(), mrd.parameters()), h.learning_rate * h.disc_lr_mult, betas=[h.adam_b1, h.adam_b2])

    validation_filelist = get_dataset_filelist(h.input_validation_wav_list)

    trainset = make_wds_audio_dataset(
        h.train_shards_path,
        h.sampling_rate,
        h.train_segment_size,
        h.sample_shuffle,
        True,
        True,
        h.seed
    )

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        batch_size=h.train_batch_size,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(h.num_workers > 0),
        prefetch_factor=2 if h.num_workers > 0 else None,
    )

    validset = Dataset(validation_filelist, h.valid_segment_size, h.sampling_rate, h.hop_size, math.prod(h.down_ratio), train=False, shuffle=False, n_cache_reuse=0, device=device)
    if world_size > 1 and dist.is_initialized():
        valid_sampler = DistributedSampler(validset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        valid_sampler = None

    validation_loader = DataLoader(
        validset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=valid_sampler,
        batch_size=h.valid_batch_size,
        pin_memory=True,
        drop_last=False,
    )

    total_steps = h.max_training_steps
    
    scheduler_g = torch.optim.lr_scheduler.OneCycleLR(
        optim_g, max_lr=h.learning_rate, total_steps=total_steps,
        pct_start=h.pct_start, div_factor=h.div_factor,
        final_div_factor=h.final_div_factor, anneal_strategy='cos',
        last_epoch=-1
    )
    scheduler_d = torch.optim.lr_scheduler.OneCycleLR(
        optim_d, max_lr=h.learning_rate * h.disc_lr_mult, total_steps=total_steps,
        pct_start=h.pct_start, div_factor=h.div_factor,
        final_div_factor=h.final_div_factor, anneal_strategy='cos',
        last_epoch=-1
    )

    if cp_state is not None:
        optim_g.load_state_dict(state_dict_state['optim_g'])
        optim_d.load_state_dict(state_dict_state['optim_d'])
        scheduler_g.load_state_dict(state_dict_state['scheduler_g'])
        scheduler_d.load_state_dict(state_dict_state['scheduler_d'])
        print(f"[Resume] Loaded scheduler states. Current lr_g={scheduler_g.get_last_lr()}, lr_d={scheduler_d.get_last_lr()}")

    sw = None
    if rank == 0:
        sw = SummaryWriter(h.logs_dir)

    encoder.train()
    quantizer.train()
    decoder.train()
    vocosbackbone.train()
    istfthead.train()
    msd.train()
    mpd.train()
    mrd.train()

    train_iter = iter(train_loader)

    while steps < h.max_training_steps:
        if h.stochastic:
            temp_steps = max(1, int(h.temp_steps))
            if steps <= temp_steps:
                current_temp = max(0.3, 1.0 - (steps / temp_steps))
            else:
                current_temp = 0.3
            _set_quantizer_mode(quantizer, stochastic=True, temperature=current_temp)
        else:
            _set_quantizer_mode(quantizer, stochastic=False, temperature=0.3)

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        start = time.time()
        y = batch
        y = torch.autograd.Variable(y.to(device, non_blocking=True))
        y = y.squeeze()
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
        if h.tail_crop_size > 0:
            y_g = y_g[..., :-h.tail_crop_size]
            y = y[..., :-h.tail_crop_size]
        mel = mel_spectrogram(y)
        mel_g = mel_spectrogram(y_g)

        # ------D step------
        if steps >= h.start_gan_steps :
            set_requires_grad(msd, True)
            set_requires_grad(mpd, True)
            set_requires_grad(mrd, True)
            optim_d.zero_grad()
            # msd
            real_score_msd, gen_score_msd, _, _ = msd(mel, mel_g.detach())
            loss_msd, loss_msd_real, _ = discriminator_loss(real_score_msd, gen_score_msd)
            loss_msd = loss_msd / len(loss_msd_real)

            # mpd
            real_score_mpd, gen_score_mpd, _, _ = mpd(y, y_g.detach())
            loss_mpd, loss_mpd_real, _ = discriminator_loss(real_score_mpd, gen_score_mpd)
            loss_mpd = loss_mpd / len(loss_mpd_real)

            # mrd
            real_score_mrd, gen_score_mrd, _, _ = mrd(y, y_g.detach())
            loss_mrd, loss_mrd_real, _ = discriminator_loss(real_score_mrd, gen_score_mrd)
            loss_mrd = loss_mrd / len(loss_mrd_real)

            L_D = loss_msd + loss_mpd + loss_mrd
            L_D.backward()
            if h.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(itertools.chain(msd.parameters(), mpd.parameters(), mrd.parameters()), max_norm=1.0)
            optim_d.step()
            scheduler_d.step()
        else:
            loss_msd = torch.tensor(0.0).to(device)
            loss_mpd = torch.tensor(0.0).to(device)
            loss_mrd = torch.tensor(0.0).to(device)
            L_D = torch.tensor(0.0).to(device)

        # ------G step------
        set_requires_grad(msd, False)
        set_requires_grad(mpd, False)
        set_requires_grad(mrd, False)
        optim_g.zero_grad()

        L_Mel_L1 = mel_loss(y_g, y)
        L_Mel = h.mel_l1_weight * L_Mel_L1 

        if steps >= h.start_gan_steps:
            if steps < h.start_gan_steps + h.gan_warmup_steps:
                disc_factor = (steps - h.start_gan_steps) / h.gan_warmup_steps
            else:
                disc_factor = 1.0

            with torch.no_grad():
                real_score_msd, _, fmap_rs_msd, _ = msd(mel, mel_g.detach())
                real_score_mpd, _, fmap_rs_mpd, _ = mpd(y, y_g.detach())
                real_score_mrd, _, fmap_rs_mrd, _ = mrd(y, y_g.detach())
            _, gen_score_msd, _, fmap_gs_msd = msd(mel, mel_g)
            _, gen_score_mpd, _, fmap_gs_mpd = mpd(y, y_g)
            _, gen_score_mrd, _, fmap_gs_mrd = mrd(y, y_g)

            loss_gen_msd, list_loss_gen_msd = generator_loss(gen_score_msd)
            loss_gen_msd = loss_gen_msd / len(list_loss_gen_msd)

            loss_gen_mpd, list_loss_gen_mpd = generator_loss(gen_score_mpd)
            loss_gen_mpd = loss_gen_mpd / len(list_loss_gen_mpd)

            loss_gen_mrd, list_loss_gen_mrd = generator_loss(gen_score_mrd)
            loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)

            loss_fm_msd = feature_loss(fmap_rs_msd, fmap_gs_msd) / len(fmap_rs_msd)
            loss_fm_mpd = feature_loss(fmap_rs_mpd, fmap_gs_mpd) / len(fmap_rs_mpd)
            loss_fm_mrd = feature_loss(fmap_rs_mrd, fmap_gs_mrd) / len(fmap_rs_mrd)

            L_GAN_G = loss_gen_msd + loss_gen_mpd + h.mrd_loss_coeff * loss_gen_mrd
            L_FM = loss_fm_msd + loss_fm_mpd + h.mrd_loss_coeff * loss_fm_mrd

            L_G = L_Mel + disc_factor * (L_GAN_G + L_FM)
            
        else:
            loss_gen_msd = torch.tensor(0.0).to(device)
            loss_gen_mpd = torch.tensor(0.0).to(device)
            loss_gen_mrd = torch.tensor(0.0).to(device)
            L_GAN_G = torch.tensor(0.0).to(device)
            
            loss_fm_msd = torch.tensor(0.0).to(device)
            loss_fm_mpd = torch.tensor(0.0).to(device)
            loss_fm_mrd = torch.tensor(0.0).to(device)
            L_FM = torch.tensor(0.0).to(device)

            L_G = L_Mel

        L_G.backward()
        if h.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(encoder.parameters(), quantizer.parameters(), decoder.parameters(), vocosbackbone.parameters(), istfthead.parameters()),
                max_norm=1.0,
            )
        optim_g.step()
        scheduler_g.step()
        set_requires_grad(msd, True)
        set_requires_grad(mpd, True)
        set_requires_grad(mrd, True)

        if rank == 0:
            if steps % h.summary_interval == 0:
                with torch.no_grad():
                    Mel_L1_error = mel_loss(y_g, y)

                print('Steps : {:d}, Gen Loss Total : {:4.3f}, Codec Mel Spectrogram L1 Loss : {:4.3f}, s/b : {:4.3f}'.
                    format(steps, L_G, Mel_L1_error, time.time() - start))

            if steps % h.summary_interval == 0:
                sw.add_scalar("train/lr", scheduler_g.get_last_lr()[0], steps)
                sw.add_scalar("Training/Generator_Total_Loss", L_G, steps)
                sw.add_scalar("Training/Codec_Mel_Loss", Mel_L1_error, steps)

                sw.add_scalar("Training/L_adv_g_MS Loss", loss_gen_msd.item(), steps)
                sw.add_scalar("Training/L_adv_g_MPD Loss", loss_gen_mpd.item(), steps)  
                sw.add_scalar("Training/L_adv_g_mrd Loss", loss_gen_mrd.item(), steps)

                sw.add_scalar("Training/L_FM_MS Loss", loss_fm_msd.item(), steps)
                sw.add_scalar("Training/L_FM_MPD Loss", loss_fm_mpd.item(), steps)
                sw.add_scalar("Training/L_FM_mrd Loss", loss_fm_mrd.item(), steps)

                sw.add_scalar("Training/MS_Loss", loss_msd.item(), steps)
                sw.add_scalar("Training/MPD_Loss", loss_mpd.item(), steps)
                sw.add_scalar("Training/mrd_Loss", loss_mrd.item(), steps)

        steps += 1

        # Validation
        if steps % h.validation_interval == 0 and steps != 0:
            encoder.eval()
            quantizer.eval()
            decoder.eval()
            vocosbackbone.eval()
            istfthead.eval()
            val_Mel_L1_err_tot = 0
            val_num = 0

            _set_quantizer_mode(quantizer, stochastic=False, temperature=0.3)

            with torch.no_grad():
                for j, batch in enumerate(validation_loader):
                    y = batch.to(device)
                    y = y.squeeze()
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
                    if h.tail_crop_size > 0:
                        y_g = y_g[..., :-h.tail_crop_size]
                        y = y[..., :-h.tail_crop_size]
                    
                    B = y.size(0)
                    val_num += B
                    val_Mel_L1_err_tot += mel_loss(y_g, y).item() * B

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                t = torch.tensor(
                    [val_Mel_L1_err_tot, float(val_num)],
                    device=device,
                    dtype=torch.float64
                )
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                val_Mel_L1_err_tot, val_num = t.tolist()
            
            if rank == 0:
                val_Mel_L1_err = val_Mel_L1_err_tot / val_num

                sw.add_scalar("Validation/Codec_Mel_loss", val_Mel_L1_err, steps)
                save_checkpoint(
                    f"{h.checkpoint_path}/codec_last",
                    {
                        'encoder': get_state_dict(encoder),
                        'quantizer': get_state_dict(quantizer),
                        'decoder': get_state_dict(decoder),
                    },
                )

                save_checkpoint(
                    f"{h.checkpoint_path}/vocos_last",
                    {'vocosbackbone': get_state_dict(vocosbackbone), 'istfthead': get_state_dict(istfthead)},
                )
                
                save_checkpoint(
                    f"{h.checkpoint_path}/state_last",
                    {
                        'msd': get_state_dict(msd),
                        'mpd': get_state_dict(mpd),
                        'mrd': get_state_dict(mrd),
                        'optim_d': optim_d.state_dict(),
                        'optim_g': optim_g.state_dict(),
                        'scheduler_d': scheduler_d.state_dict(),
                        'scheduler_g': scheduler_g.state_dict(),
                        'steps': steps,
                    },
                )

        encoder.train()
        quantizer.train()
        decoder.train()
        vocosbackbone.train()
        istfthead.train()


def main():
    print('Initializing Training Process..')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default="", help='Path to config YAML')
    parser.add_argument('--resume_from_checkpoint', type=str, default="", help='Path to codec_* or vocos_* or state_* checkpoint')
    args = parser.parse_args()
    config_file = args.config
    h = load_hparams(config_file)

    is_distributed = ('RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ)

    if is_distributed:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        print("="*80)
        print(f"[Distributed Setup] Hostname: {socket.gethostname()}")
        print(f"[Distributed Setup] RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
        print(f"[Distributed Setup] MASTER_ADDR={os.environ.get('MASTER_ADDR', 'NOT SET')}")
        print(f"[Distributed Setup] MASTER_PORT={os.environ.get('MASTER_PORT', 'NOT SET')}")
        print(f"[Distributed Setup] NODE_RANK={os.environ.get('NODE_RANK', 'NOT SET')}")
        print(f"[Distributed Setup] NODE_COUNT={os.environ.get('NODE_COUNT', 'NOT SET')}")
        print("="*80)

        backend = h.ddp_backend
        dist.init_process_group(backend=backend, init_method='env://', world_size=world_size, rank=rank)
    else:
        rank, world_size, local_rank = 0, 1, 0

    exp_name = h.exp_name
    runs_root = h.runs_root
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    run_id = h.run_id or f"{timestamp}__seed{h.seed}__gpu{world_size}"
    run_dir = os.path.join(runs_root, exp_name, run_id)

    if is_distributed:
        obj_list = [run_dir if rank == 0 else None]
        dist.broadcast_object_list(obj_list, src=0)
        run_dir = obj_list[0]

    meta_dir = os.path.join(run_dir, 'meta')
    logs_dir = os.path.join(run_dir, 'logs')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    samples_dir = os.path.join(run_dir, 'samples')
    metrics_dir = os.path.join(run_dir, "metrics")

    if rank == 0:
        os.makedirs(meta_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        try:
            build_env(config_file, 'config.yaml', meta_dir)
        except Exception:
            import shutil
            shutil.copyfile(config_file, os.path.join(meta_dir, 'config.yaml'))

        with open(os.path.join(meta_dir, 'config_resolved.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(dict(h), f, allow_unicode=True, sort_keys=False)

        with open(os.path.join(meta_dir, 'env.txt'), 'w', encoding='utf-8') as f:
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


if __name__ == '__main__':
    main()
