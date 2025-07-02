<div align="center">

# ZipVoice⚡

## Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching

[![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](http://arxiv.org/abs/2506.13053)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://zipvoice.github.io/)
</div>

## Overview

ZipVoice is a high-quality zero-shot TTS model with a small model size and fast inference speed.

### 1. Key features

- Small and fast: only 123M parameters.

- High-quality: state-of-the-art voice cloning performance in speaker similarity, intelligibility, and naturalness.

- Multi-lingual: support Chinese and English.

### 2. Architecture

<div align="center">

<img src="https://zipvoice.github.io/pics/zipvoice.png" width="700" >

</div>

## News

**2025/06/16**: 🔥 ZipVoice is released.

## Installation

### 1. Clone the ZipVoice repository

```bash
git clone https://github.com/k2-fsa/ZipVoice.git
```

### 2. (Optional) Create a Python virtual environment

```bash
python3 -m venv zipvoice
source zipvoice/bin/activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install k2 for training or efficient inference:

k2 is necessary for training and can speed up inference. Nevertheless, you can still use the inference mode of ZipVoice without installing k2.

> **Note:**  Make sure to install the k2 version that matches your PyTorch and CUDA version. For example, if you are using pytorch 2.5.1 and CUDA 12.1, you can install k2 as follows:

```bash
pip install k2==1.24.4.dev20250208+cuda12.1.torch2.5.1 -f https://k2-fsa.github.io/k2/cuda.html
```

Please refer to https://k2-fsa.org/get-started/k2/ for details.
Users in China mainland can refer to https://k2-fsa.org/zh-CN/get-started/k2/.

## Usage

To generate speech with our pre-trained ZipVoice or ZipVoice-Distill models, use the following commands (Required models will be downloaded from HuggingFace):

### 1. Inference of a single sentence

```bash
python3 zipvoice/zipvoice_infer.py \
    --model-name "zipvoice" \
    --prompt-wav prompt.wav \
    --prompt-text "I am the transcription of the prompt wav." \
    --text "I am the text to be synthesized." \
    --res-wav-path result.wav
```

- `--model-name` can be `zipvoice` or `zipvoice_distill`, which are models before and after distillation, respectively.
- If `<>` or `[]` appear in the text, strings enclosed by them will be treated as special tokens. `<>` denotes Chinese pinyin and `[]` denotes other special tags.

### 2. Inference of a list of sentences

```bash
python3 zipvoice/zipvoice_infer.py \
    --model-name "zipvoice" \
    --test-list test.tsv \
    --res-dir results/test
```

- Each line of `test.tsv` is in the format of `{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}`.

> Could run ONNX models for faster inference on CPU with `zipvoice/zipvoice_infer_onnx.py`.

> **Note:** If you have trouble connecting to HuggingFace, try:
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

### 3. Correcting mispronounced chinese polyphone characters

We use [pypinyin](https://github.com/mozillazg/python-pinyin) to convert Chinese characters to pinyin. However, it can occasionally mispronounce **polyphone characters** (多音字).

To manually correct these mispronunciations, enclose the **corrected pinyin** in angle brackets `< >` and include the **tone mark**.

**Example:**

- Original text: `这把剑长三十公分`
- Correct the pinyin of `长`:  `这把剑<chang2>三十公分`

> **Note:** If you want to manually assign multiple pinyins, enclose each pinyin with `<>`, e.g., `这把<jian4><chang2><san1>十公分`

## Training Your Own Model

The following steps show how to train a model from scratch on Emilia and LibriTTS datasets, respectively.

### 1. Data Preparation

#### 1.1. Prepare the Emilia dataset

```bash
bash scripts/prepare_emilia.sh
```

See [scripts/prepare_emilia.sh](scripts/prepare_emilia.sh) for step by step instructions.

#### 1.2 Prepare the LibriTTS dataset

```bash
bash scripts/prepare_libritts.sh
```

See [scripts/prepare_libritts.sh](scripts/prepare_libritts.sh) for step by step instructions.

### 2. Training

#### 2.1 Traininig on Emilia

<details>
<summary>Expand to view training steps</summary>

##### 2.1.1 Train the ZipVoice model

- Training:

```bash
python3 zipvoice/train_flow.py \
        --world-size 8 \
        --use-fp16 1 \
        --dataset emilia \
        --max-duration 500 \
        --lr-hours 30000 \
        --lr-batches 7500 \
        --token-file "data/tokens_emilia.txt" \
        --manifest-dir "data/fbank" \
        --num-epochs 11 \
        --exp-dir zipvoice/exp_zipvoice
```

- Average the checkpoints to produce the final model:

```bash
python3 zipvoice/generate_averaged_model.py \
      --epoch 11 \
      --avg 4 \
      --distill 0 \
      --token-file data/tokens_emilia.txt \
      --dataset "emilia" \
      --exp-dir ./zipvoice/exp_zipvoice
# The generated model is zipvoice/exp_zipvoice/epoch-11-avg-4.pt
```

##### 2.1.2. Train the ZipVoice-Distill model (Optional)

- The first-stage distillation:

```bash
python3 zipvoice/train_distill.py \
        --world-size 8 \
        --use-fp16 1 \
        --tensorboard 1 \
        --dataset "emilia" \
        --base-lr 0.0005 \
        --max-duration 500 \
        --token-file "data/tokens_emilia.txt" \
        --manifest-dir "data/fbank" \
        --teacher-model zipvoice/exp_zipvoice/epoch-11-avg-4.pt \
        --num-updates 60000 \
        --distill-stage "first" \
        --exp-dir zipvoice/exp_zipvoice_distill_1stage
```

- Average checkpoints for the second-stage initialization:

```bash
python3 zipvoice/generate_averaged_model.py \
      --iter 60000 \
      --avg 7 \
      --distill 1 \
      --token-file data/tokens_emilia.txt \
      --dataset "emilia" \
      --exp-dir ./zipvoice/exp_zipvoice_distill_1stage
# The generated model is zipvoice/exp_zipvoice_distill_1stage/iter-60000-avg-7.pt
```

- The second-stage distillation:

```bash
python3 zipvoice/train_distill.py \
        --world-size 8 \
        --use-fp16 1 \
        --tensorboard 1 \
        --dataset "emilia" \
        --base-lr 0.0001 \
        --max-duration 200 \
        --token-file "data/tokens_emilia.txt" \
        --manifest-dir "data/fbank" \
        --teacher-model zipvoice/exp_zipvoice_distill_1stage/iter-60000-avg-7.pt \
        --num-updates 2000 \
        --distill-stage "second" \
        --exp-dir zipvoice/exp_zipvoice_distill_new
```

</details>

#### 2.2 Traininig on LibriTTS

<details>
<summary>Expand to view training steps</summary>

##### 2.2.1 Train the ZipVoice model

- Training:

```bash
python3 zipvoice/train_flow.py \
        --world-size 8 \
        --use-fp16 1 \
        --dataset libritts \
        --token-type char \
        --max-duration 250 \
        --lr-epochs 10 \
        --lr-batches 7500 \
        --token-file "data/tokens_libritts.txt" \
        --manifest-dir "data/fbank" \
        --num-epochs 60 \
        --exp-dir zipvoice/exp_zipvoice_libritts
```

- Average the checkpoints to produce the final model:

```bash
python3 zipvoice/generate_averaged_model.py \
      --epoch 60 \
      --avg 10 \
      --distill 0 \
      --token-file data/tokens_libritts.txt \
      --dataset "libritts" \
      --exp-dir ./zipvoice/exp_zipvoice_libritts
# The generated model is zipvoice/exp_zipvoice_libritts/epoch-60-avg-10.pt
```

##### 2.1.2 Train the ZipVoice-Distill model (Optional)

- The first-stage distillation:

```bash
python3 zipvoice/train_distill.py \
        --world-size 8 \
        --use-fp16 1 \
        --tensorboard 1 \
        --dataset "libritts" \
        --base-lr 0.001 \
        --max-duration 250 \
        --token-file "data/tokens_libritts.txt" \
        --manifest-dir "data/fbank" \
        --teacher-model zipvoice/exp_zipvoice_libritts/epoch-60-avg-10.pt \
        --num-epochs 6 \
        --distill-stage "first" \
        --exp-dir zipvoice/exp_zipvoice_distill_1stage_libritts
```

- Average checkpoints for the second-stage initialization:

```bash
python3 ./zipvoice/generate_averaged_model.py \
      --epoch 6 \
      --avg 3 \
      --distill 1 \
      --token-file data/tokens_libritts.txt \
      --dataset "libritts" \
      --exp-dir ./zipvoice/exp_zipvoice_distill_1stage_libritts
# The generated model is zipvoice/exp_zipvoice_distill_1stage_libritts/epoch-6-avg-3.pt
```

- The second-stage distillation:

```bash
python3 zipvoice/train_distill.py \
        --world-size 8 \
        --use-fp16 1 \
        --tensorboard 1 \
        --dataset "libritts" \
        --base-lr 0.001 \
        --max-duration 250 \
        --token-file "data/tokens_libritts.txt" \
        --manifest-dir "data/fbank" \
        --teacher-model zipvoice/exp_zipvoice_distill_1stage_libritts/epoch-6-avg-3.pt \
        --num-epochs 6 \
        --distill-stage "second" \
        --exp-dir zipvoice/exp_zipvoice_distill_libritts
```

- Average checkpoints to produce the final model:

```bash
python3 ./zipvoice/generate_averaged_model.py \
      --epoch 6 \
      --avg 3 \
      --distill 1 \
      --token-file data/tokens_libritts.txt \
      --dataset "libritts" \
      --exp-dir ./zipvoice/exp_zipvoice_distill_libritts
# The generated model is ./zipvoice/exp_zipvoice_distill_libritts/epoch-6-avg-3.pt
```

</details>

### 3. Inference with the trained model

#### 3.1 Inference with the model trained on Emilia

<details>
<summary>Expand to view inference commands.</summary>

##### 3.1.1 ZipVoice model before distill

```bash
python3 zipvoice/infer.py \
      --checkpoint zipvoice/exp_zipvoice/epoch-11-avg-4.pt \
      --distill 0 \
      --token-file "data/tokens_emilia.txt" \
      --test-list test.tsv \
      --res-dir results/test \
      --num-step 16 \
      --guidance-scale 1
```

##### 3.1.2 ZipVoice-Distill model

```bash
python3 zipvoice/infer.py \
      --checkpoint zipvoice/exp_zipvoice_distill/checkpoint-2000.pt \
      --distill 1 \
      --token-file "data/tokens_emilia.txt" \
      --test-list test.tsv \
      --res-dir results/test_distill \
      --num-step 8 \
      --guidance-scale 3
```

</details>

#### 3.2 Inference with the model trained on LibriTTS

<details>
<summary>Expand to view inference commands.</summary>

##### 3.2.1 ZipVoice model before distill

```bash
python3 zipvoice/infer.py \
      --checkpoint zipvoice/exp_zipvoice_libritts/epoch-60-avg-10.pt \
      --distill 0 \
      --token-file "data/tokens_libritts.txt" \
      --test-list test.tsv \
      --res-dir results/test_libritts \
      --num-step 8 \
      --guidance-scale 1 \
      --target-rms 1.0 \
      --t-shift 0.7
```

##### 3.2.2 ZipVoice-Distill model

```bash
python3 zipvoice/infer.py \
      --checkpoint zipvoice/exp_zipvoice_distill/epoch-6-avg-3.pt \
      --distill 1 \
      --token-file "data/tokens_libritts.txt" \
      --test-list test.tsv \
      --res-dir results/test_distill_libritts \
      --num-step 4 \
      --guidance-scale 3 \
      --target-rms 1.0 \
      --t-shift 0.7
```

</details>

### 4. Evaluation on benchmarks

See [local/evaluate.sh](local/evaluate.sh) for details of objective metrics evaluation
on three test sets, i.e., LibriSpeech-PC test-clean, Seed-TTS test-en and Seed-TTS test-zh.

## Discussion & Communication

You can directly discuss on [Github Issues](https://github.com/k2-fsa/ZipVoice/issues).

You can also scan the QR code to join our wechat group or follow our wechat official account.

| Wechat Group | Wechat Official Account |
| ------------ | ----------------------- |
|![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_group.jpg) |![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_account.jpg) |

## Citation

```bibtex
@article{zhu2025zipvoice,
      title={ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching},
      author={Zhu, Han and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Li, Zhaoqing and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2506.13053},
      year={2025}
}
```
