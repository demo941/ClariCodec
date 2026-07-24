<div align="center">

# ClariCodec

### Optimising Neural Speech Codecs for 300bps Communication using Reinforcement Learning

Junyi Wang, Chi Zhang, Jing Qian, Haifeng Luo, Hao Wang, Zengrui Jin, and Chao Zhang

<p>
  <a href="https://doi.org/10.48550/arXiv.2605.19541"><img src="https://img.shields.io/badge/arXiv-2605.19541-b31b1b.svg" alt="arXiv"></a>
  <a href="https://demo941.github.io/claricodec-demo-page/"><img src="https://img.shields.io/badge/Demo-Project_Page-2ea44f.svg" alt="Demo"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
</p>

</div>

ClariCodec is a 300 bit-per-second neural speech codec designed for intelligibility-critical communication under severe bandwidth constraints. It treats quantisation as a stochastic policy and fine-tunes the encoder with word-error-rate rewards while keeping the acoustic reconstruction pipeline frozen.

This repository contains non-streaming and streaming implementations for first-stage codec training, reinforcement-learning fine-tuning, inference, and evaluation. Datasets, ASR models, and codec checkpoints are not distributed in this repository and must be prepared separately.

## Highlights

- Ultra-low-bitrate speech coding at 300 bps.
- WER-driven reinforcement learning for intelligibility optimisation.
- Non-streaming and streaming codec implementations.
- Local or remotely streamed WebDataset training shards.
- Checkpoint resume support for models, optimisers, and learning-rate schedulers.

The paper reports 4.64% WER on LibriSpeech test-clean before RL fine-tuning. RL fine-tuning reduces WER to 3.55% on test-clean and 10.4% on test-other, corresponding to a 23% relative reduction on test-clean while preserving perceptual quality.

## Repository Structure

```text
ClariCodec/
├── configs/
│   ├── base.yaml
│   └── streaming.yaml
├── src/
│   ├── codec/
│   ├── data/
│   ├── loss/
│   ├── utils/
│   └── vocos/
└── tools/
    ├── train.py
    ├── train_streaming.py
    ├── rl.py
    ├── rl_streaming.py
    ├── inference.py
    ├── inference_streaming.py
    ├── cal_metrics.py
    └── cal_wer.py
```

## Installation

The code was developed with the following main environment:

| Component | Version |
|---|---:|
| Python | 3.11.14 |
| PyTorch | 2.7.1 |
| torchaudio | 2.7.1 |
| CUDA | 12.8 |
| NVIDIA NeMo | 2.6.0 |

Create an isolated environment and install the direct Python dependencies:

```bash
conda create -n claricodec python=3.11 -y
conda activate claricodec
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA driver if it differs from the environment above. Remote WebDataset streaming additionally requires `rclone`. WER evaluation through NeMo may require system audio tools such as FFmpeg and libsndfile.

## Data Preparation

### Training shards

Training uses WebDataset tar shards. Set `train_shards_path` in the selected YAML configuration to a text file containing one shard per line:

```text
/path/to/shard-000000.tar
/path/to/shard-000001.tar
```

Remote shards accessible through rclone are also supported:

```text
remote_name:path/to/shard-000000.tar
remote_name:path/to/shard-000001.tar
```

Each shard should contain audio samples encoded as WAV or FLAC.

### Validation data

Set `input_validation_wav_list` to a text file containing one audio path per line:

```text
/path/to/validation/0001.wav
/path/to/validation/0002.wav
```

For WER evaluation, the reference transcription file must use the following format:

```text
REFERENCE TRANSCRIPTION (utterance_id)
```

Update all `path/to/...` entries in `configs/base.yaml` or `configs/streaming.yaml` before running an experiment.

## First-Stage Training

Run non-streaming training on one GPU:

```bash
python tools/train.py --config configs/base.yaml
```

Run streaming training on one GPU:

```bash
python tools/train_streaming.py --config configs/streaming.yaml
```

For distributed training, launch the same entry points with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 tools/train.py --config configs/base.yaml
torchrun --standalone --nproc_per_node=8 tools/train_streaming.py --config configs/streaming.yaml
```

Replace `8` with the number of GPUs. Experiment outputs are written below `runs_root/exp_name/run_id`. Each validation cycle updates `codec_last`, `vocos_last`, and `state_last`.

Resume from any member of a matching checkpoint triplet:

```bash
python tools/train.py \
  --config configs/base.yaml \
  --resume_from_checkpoint runs/<experiment>/<run>/checkpoints/state_last
```

## Reinforcement-Learning Fine-Tuning

Before RL fine-tuning:

1. Set `rl.checkpoint_path_codec` and `rl.checkpoint_path_vocos` in the selected configuration.
2. Set the NeMo ASR checkpoint used by `Parakeet` in `src/codec/asr_model.py`.
3. Verify the RL batch size, group size, learning rate, and temperature.

Run non-streaming RL fine-tuning:

```bash
python tools/rl.py --config configs/base.yaml
```

Run streaming RL fine-tuning:

```bash
python tools/rl_streaming.py --config configs/streaming.yaml
```

The RL validation loop reports mel reconstruction loss and WER. Resume training with the same `--resume_from_checkpoint` convention used by first-stage training.

## Inference

Set `checkpoint_file_load_Codec`, `checkpoint_file_load_Vocos`, `test_input_wavs_dir`, and `test_wav_output_dir` in the selected configuration.

Non-streaming inference:

```bash
python tools/inference.py --config configs/base.yaml
```

Streaming inference:

```bash
python tools/inference_streaming.py --config configs/streaming.yaml
```

The inference scripts also report codebook statistics and the estimated bitrate.

## Evaluation

Calculate STOI and PESQ on generated audio:

```bash
python tools/cal_metrics.py --config configs/base.yaml
```

Calculate WER with a NeMo ASR checkpoint:

```bash
python tools/cal_wer.py \
  --config configs/base.yaml \
  --model_path /path/to/asr_model.nemo
```

Evaluation paths and output logs are configured in the `Metrics` section of each YAML file.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{wang2026claricodec,
  title={Optimising Neural Speech Codecs for 300bps Communication using Reinforcement Learning},
  author={Wang, Junyi and Zhang, Chi and Qian, Jing and Luo, Haifeng and Wang, Hao and Jin, Zengrui and Zhang, Chao},
  journal={arXiv preprint arXiv:2605.19541},
  year={2026},
  doi={10.48550/arXiv.2605.19541}
}
```

## Acknowledgements

This project builds on ideas and components from [Vocos](https://github.com/gemelo-ai/vocos), [finite scalar quantisation](https://github.com/google-deepmind/finite_scalar_quantization), [WebDataset](https://github.com/webdataset/webdataset), and [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).

## License

ClariCodec is released under the [MIT License](LICENSE).
