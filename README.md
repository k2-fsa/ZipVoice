
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>
      <div align="center">
        <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=en">English</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=zh-CN">简体中文</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=zh-TW">繁體中文</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=ja">日本語</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=ko">한국어</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=hi">हिन्दी</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=th">ไทย</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=fr">Français</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=de">Deutsch</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=es">Español</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=it">Itapano</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=ru">Русский</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=pt">Português</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=nl">Nederlands</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=pl">Polski</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=ar">العربية</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=fa">فارسی</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=tr">Türkçe</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=vi">Tiếng Việt</a>
        | <a href="https://openaitx.github.io/view.html?user=k2-fsa&project=ZipVoice&lang=id">Bahasa Indonesia</a>
      </div>
    </div>
  </details>
</div>

<div align="center">

# ZipVoice⚡

## Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching
</div>

## Overview

ZipVoice is a series of fast and high-quality zero-shot TTS models based on flow matching.

### 1. Key features

- Small and fast: only 123M parameters.

- High-quality voice cloning: state-of-the-art performance in speaker similarity, intelligibility, and naturalness.

- Multi-lingual: support Chinese and English.

- Multi-mode: support both single-speaker and dialogue speech generation.

### 2. Model variants

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Description</th>
      <th>Paper</th>
      <th>Demo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ZipVoice</td>
      <td>The basic model supporting zero-shot single-speaker TTS in both Chinese and English.</td>
      <td rowspan="2"><a href="https://arxiv.org/abs/2506.13053"><img src="https://img.shields.io/badge/arXiv-Paper-COLOR.svg"></a></td>
      <td rowspan="2"><a href="https://zipvoice.github.io"><img src="https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square"></a></td>
    </tr>
    <tr>
      <td>ZipVoice-Distill</td>
      <td>The distilled version of ZipVoice, featuring improved speed with minimal performance degradation.</td>
    </tr>
    <tr>
      <td>ZipVoice-Dialog</td>
      <td>A dialogue generation model built on ZipVoice, capable of generating single-channel two-party spoken dialogues.</td>
      <td rowspan="2"><a href="https://arxiv.org/abs/2507.09318"><img src="https://img.shields.io/badge/arXiv-Paper-COLOR.svg"></a></td>
      <td rowspan="2"><a href="https://zipvoice-dialog.github.io"><img src="https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square"></a></td>
    </tr>
    <tr>
      <td>ZipVoice-Dialog-Stereo</td>
      <td>The stereo variant of ZipVoice-Dialog, enabling two-channel dialogue generation with each speaker assigned to a distinct channel.</td>
    </tr>
  </tbody>
</table>

## News

**2025/07/14**: **ZipVoice-Dialog** and **ZipVoice-Dialog-Stereo**, two spoken dialogue generation models, are released. [![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2507.09318) [![demo page](https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square)](https://zipvoice-dialog.github.io)

**2025/07/14**: **OpenDialog** dataset, a 6.8k-hour spoken dialogue dataset, is realeased. Download at [![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/k2-fsa/OpenDialog), [![ms](https://img.shields.io/badge/ModelScope-Dataset-blue?logo=data)](https://www.modelscope.cn/datasets/k2-fsa/OpenDialog). Check details at [![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2507.09318).

**2025/06/16**: **ZipVoice** and **ZipVoice-Distill** are released. [![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2506.13053) [![demo page](https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square)](https://zipvoice.github.io)

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

### 4. Install k2 for training or efficient inference

**k2 is necessary for training** and can speed up inference. Nevertheless, you can still use the inference mode of ZipVoice without installing k2.

> **Note:**  Make sure to install the k2 version that matches your PyTorch and CUDA version. For example, if you are using pytorch 2.5.1 and CUDA 12.1, you can install k2 as follows:

```bash
pip install k2==1.24.4.dev20250208+cuda12.1.torch2.5.1 -f https://k2-fsa.github.io/k2/cuda.html
```

Please refer to https://k2-fsa.org/get-started/k2/ for details.
Users in China mainland can refer to https://k2-fsa.org/zh-CN/get-started/k2/.

- To check the k2 installation:

```
python3 -c "import k2; print(k2.__file__)"
```

## Usage

### 1. Single-speaker speech generation

To generate single-speaker speech with our pre-trained ZipVoice or ZipVoice-Distill models, use the following commands (Required models will be downloaded from HuggingFace):

#### 1.1 Inference of a single sentence

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
- Could run ONNX models on CPU faster with `zipvoice.bin.infer_zipvoice_onnx`.

> **Note:** If you have trouble connecting to HuggingFace, try:
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

#### 1.2 Inference of a list of sentences

```bash
python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --test-list test.tsv \
    --res-dir results
```

- Each line of `test.tsv` is in the format of `{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}`.

### 2. Dialogue speech generation

#### 2.1 Inference command

To generate two-party spoken dialogues with our pre-trained ZipVoice-Dialogue or ZipVoice-Dialogue-Stereo models, use the following commands (Required models will be downloaded from HuggingFace):

```bash
python3 -m zipvoice.bin.infer_zipvoice_dialog \
    --model-name "zipvoice_dialog" \
    --test-list test.tsv \
    --res-dir results
```

- `--model-name` can be `zipvoice_dialog` or `zipvoice_dialog_stereo`,
    which generate mono and stereo dialogues, respectively.

#### 2.2 Input formats

Each line of `test.tsv` is in one of the following formats:

(1) **Merged prompt format** where the audios and transcriptions of two speakers prompts are merged into one prompt wav file:
```
{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}
```

- `wav_name` is the name of the output wav file.
- `prompt_transcription` is the transcription of the conversational prompt wav, e.g, "[S1] Hello. [S2] How are you?"
- `prompt_wav` is the path to the prompt wav.
- `text` is the text to be synthesized, e.g. "[S1] I'm fine. [S2] What's your name?"

(2) **Splitted prompt format** where the audios and transciptions of two speakers exist in separate files:

```
{wav_name}\t{spk1_prompt_transcription}\t{spk2_prompt_transcription}\t{spk1_prompt_wav}\t{spk2_prompt_wav}\t{text}'
```

- `wav_name` is the name of the output wav file.
- `spk1_prompt_transcription` is the transcription of the first speaker's prompt wav, e.g, "Hello"
- `spk2_prompt_transcription` is the transcription of the second speaker's prompt wav, e.g, "How are you?"
- `spk1_prompt_wav` is the path to the first speaker's prompt wav file.
- `spk2_prompt_wav` is the path to the second speaker's prompt wav file.
- `text` is the text to be synthesized, e.g. "[S1] I'm fine. [S2] What's your name?"

### 3. Other features

#### 3.1 Correcting mispronounced chinese polyphone characters

We use [pypinyin](https://github.com/mozillazg/python-pinyin) to convert Chinese characters to pinyin. However, it can occasionally mispronounce **polyphone characters** (多音字).

To manually correct these mispronunciations, enclose the **corrected pinyin** in angle brackets `< >` and include the **tone mark**.

**Example:**

- Original text: `这把剑长三十公分`
- Correct the pinyin of `长`:  `这把剑<chang2>三十公分`

> **Note:** If you want to manually assign multiple pinyins, enclose each pinyin with `<>`, e.g., `这把<jian4><chang2><san1>十公分`

## Train Your Own Model

See the [egs](egs) directory for training, fine-tuning and evaluation examples.

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

@article{zhu2025zipvoicedialog,
      title={ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching},
      author={Zhu, Han and Kang, Wei and Guo, Liyong and Yao, Zengwei and Kuang, Fangjun and Zhuang, Weiji and Li, Zhaoqing and Han, Zhifeng and Zhang, Dong and Zhang, Xin and Song, Xingchen and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2507.09318},
      year={2025}
}
```
