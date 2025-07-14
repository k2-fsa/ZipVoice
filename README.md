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
python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --prompt-wav prompt.wav \
    --prompt-text "I am the transcription of the prompt wav." \
    --text "I am the text to be synthesized." \
    --res-wav-path result.wav
```

- `--model-name` can be `zipvoice` or `zipvoice_distill`, which are models before and after distillation, respectively.
- If `<>` or `[]` appear in the text, strings enclosed by them will be treated as special tokens. `<>` denotes Chinese pinyin and `[]` denotes other special tags.

### 2. Inference of a list of sentences

```bash
python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --test-list test.tsv \
    --res-dir results/test
```

- Each line of `test.tsv` is in the format of `{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}`.

> Could run ONNX models on CPU faster with `zipvoice.bin.infer_zipvoice_onnx`.

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

See [examples](egs) for training examples.

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
