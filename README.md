<div align="center">

# ClariCodec

### Optimising Neural Speech Codecs for 300 bps Communication Using Reinforcement Learning

Junyi Wang, Chi Zhang, Jing Qian, Haifeng Luo, Hao Wang, Zengrui Jin, and Chao Zhang

<p>
  <a href="https://doi.org/10.48550/arXiv.2605.19541"><img src="https://img.shields.io/badge/arXiv-2605.19541-b31b1b.svg" alt="arXiv"></a>
  <a href="https://demo941.github.io/claricodec-demo-page/"><img src="https://img.shields.io/badge/Demo-Project_Page-2ea44f.svg" alt="Demo"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
</p>

</div>

ClariCodec is a neural speech codec designed for intelligibility-critical communication under severe bandwidth constraints. It operates at 300 bits per second (bps), reformulates quantisation as a stochastic policy, and uses Group Relative Policy Optimisation (GRPO) with word-error-rate rewards to fine-tune the encoder while keeping the acoustic reconstruction pipeline frozen.

This repository contains non-streaming and streaming implementations for first-stage codec training, reinforcement-learning fine-tuning, inference, and evaluation. 

## Highlights

- At 300 bps, ClariCodec achieves 4.64% WER on LibriSpeech test-clean before RL fine-tuning.
- RL fine-tuning reduces test-clean WER to 3.55%, a 23.5% relative reduction, while preserving perceptual quality.
- The streaming model achieves 4.53% WER on test-clean with a theoretical latency of 374 ms.

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

## Pretrained Checkpoints

The paper was accepted at **SLT 2026**. We release all four 300 bps models used in the main comparisons. The RL models are recommended for inference; the Stage 1 models are provided for comparison and further RL fine-tuning.

| Model | LibriSpeech test-clean WER | Theoretical latency | Download | Configuration |
|---|---:|---:|---|---|
| Non-streaming Stage 1 | 4.64% | Offline | [Checkpoint](https://github.com/demo941/ClariCodec/releases/download/v1.0.0/claricodec-300bps-nonstreaming-stage1.tar) | [YAML](configs/pretrained/nonstreaming-stage1.yaml) |
| **Non-streaming RL** | **3.55%** | Offline | [Checkpoint](https://github.com/demo941/ClariCodec/releases/download/v1.0.0/claricodec-300bps-nonstreaming-rl.tar) | [YAML](configs/pretrained/nonstreaming-rl.yaml) |
| Streaming Stage 1 | 6.63% | 374 ms | [Checkpoint](https://github.com/demo941/ClariCodec/releases/download/v1.0.0/claricodec-300bps-streaming-stage1.tar) | [YAML](configs/pretrained/streaming-stage1.yaml) |
| **Streaming RL** | **4.53%** | **374 ms** | [Checkpoint](https://github.com/demo941/ClariCodec/releases/download/v1.0.0/claricodec-300bps-streaming-rl.tar) | [YAML](configs/pretrained/streaming-rl.yaml) |

Each archive contains `codec.pt` (encoder, complete quantizer, and decoder), `vocos.pt` (Vocos backbone and ISTFT head), `config.yaml`, model metadata, and the MIT license. Weights retain their original FP32 precision. Optimizer and discriminator states are not included. Checkpoint step counts in the metadata refer to the corresponding training stage.

Download one model, or all four:

```bash
python tools/download_checkpoints.py --model nonstreaming-rl
python tools/download_checkpoints.py --model streaming-rl
# Optional: download the Stage 1 models as well.
python tools/download_checkpoints.py --model all
```

The downloader extracts each model into `checkpoints/<model-name>/`. Run the following commands from the repository root after setting `test_input_wavs_dir` in the selected YAML to a directory of input WAV files:

```bash
python tools/inference.py --config configs/pretrained/nonstreaming-rl.yaml
python tools/inference_streaming.py --config configs/pretrained/streaming-rl.yaml
```

For Stage 1 inference, select the corresponding `nonstreaming-stage1.yaml` or `streaming-stage1.yaml`. The bundled `checkpoints/<model-name>/config.yaml` can also be used directly. Audio is mono at 16 kHz, with a 12.5 Hz code-frame rate and 300 bps bitrate. Inference runs locally without downloading an ASR model.

## Data Preparation

Training uses WebDataset tar shards. Set `train_shards_path` in the selected YAML configuration to a text file containing one local or rclone-accessible shard per line:

```text
/path/to/shard-000000.tar
/path/to/shard-000001.tar
remote_name:path/to/shard-000000.tar
```

Each shard should contain audio samples encoded as WAV or FLAC.

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

## Usage

### First-stage training

Use `base.yaml` for the non-streaming model and `streaming.yaml` for the streaming model:

```bash
python tools/train.py --config configs/base.yaml
python tools/train_streaming.py --config configs/streaming.yaml
```

### Reinforcement-learning fine-tuning

Set the codec, Vocos, and NeMo ASR checkpoint paths before running:

```bash
python tools/rl.py --config configs/base.yaml
python tools/rl_streaming.py --config configs/streaming.yaml
```

### Inference

Set `checkpoint_file_load_Codec`, `checkpoint_file_load_Vocos`, `test_input_wavs_dir`, and `test_wav_output_dir` in the selected configuration.

```bash
python tools/inference.py --config configs/base.yaml
python tools/inference_streaming.py --config configs/streaming.yaml
```

### Evaluation

Run objective quality metrics and WER evaluation as follows:

```bash
python tools/cal_metrics.py --config configs/base.yaml
python tools/cal_wer.py --config configs/base.yaml --model_path /path/to/asr_model.nemo
```

The training scripts support distributed launch with `torchrun` and resuming with `--resume_from_checkpoint`.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{wang2026claricodec,
  title={Optimising Neural Speech Codecs for 300 bps Communication Using Reinforcement Learning},
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
