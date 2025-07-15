#!/bin/bash

# This script is an example of fine-tuning ZipVoice on your custom datasets.

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=6

# Number of jobs for data preparation
nj=20

# Whether the language of training data is one of Chinese and English
is_zh_en=1

# Language identifier, used when language is not Chinese or English
# see https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md
lang=en-us

# You can set `max_len` according to statistics from the command 
# `lhotse cut describe data/fbank/custom_cuts_train.jsonl.gz`.
# Set `max_len` to 99% duration.

# Maximum length (seconds) of the training utterance, will filter out longer utterances
max_len=20

# Download directory for pre-trained models
download_dir=download/

# We suppose you have two TSV files: "data/raw/custom_train.tsv" and 
# "data/raw/custom_dev.tsv", where "custom" is your dataset name, 
# "train"/"dev" are used for training and validation respectively.

# Each line of the TSV files should be in one of the following formats:
# (1) `{uniq_id}\t{text}\t{wav_path}` if the text corresponds to the full wav,
# (2) `{uniq_id}\t{text}\t{wav_path}\t{start_time}\t{end_time}` if text corresponds
#     to part of the wav. The start_time and end_time specify the start and end
#     times of the text within the wav, which should be in seconds.
# > Note: {uniq_id} must be unique for each line.
for subset in train dev;do
      file_path=data/raw/custom_${subset}.tsv
      [ -f "$file_path" ] || { echo "Error: expect $file_path !" >&2; exit 1; }
done

### Prepare the training data (1 - 3)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Prepare manifests for custom dataset from tsv files"

      for subset in train dev;do
            python3 local/prepare_custom_dataset.py \
                  --tsv-path data/raw/custom_${subset}.tsv \
                  --prefix custom \
                  --subset ${subset} \
                  --num-jobs ${nj} \
                  --output-dir data/manifests
      done
      # The output manifest files are "data/manifests/custom_cuts_train.jsonl.gz".
      # and "data/manifests/custom_cuts_dev.jsonl.gz".
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
      echo "Stage 2: Compute Fbank for custom dataset"
      # You can skip this step and use `--on-the-fly-feats 1` in training stage
      for subset in train dev; do
            python3 -m zipvoice.bin.compute_fbank \
                  --source-dir data/manifests \
                  --dest-dir data/fbank \
                  --dataset custom \
                  --subset ${subset} \
                  --num-jobs ${nj}
      done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Download pre-trained model, tokens file, and model config"
      # Uncomment this line to use HF mirror
      # export HF_ENDPOINT=https://hf-mirror.com
      hf_repo=zhu-han/ZipVoice
      mkdir -p ${download_dir}
      for file in model.pt tokens.txt zipvoice_base.json; do
            huggingface-cli download \
                  --local-dir ${download_dir} \
                  ${hf_repo} \
                  zipvoice/${file}
      done
fi

### Training ZipVoice (4 - 5)

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Fine-tune the ZipVoice model"

      [ -z "$max_len" ] && { echo "Error: max_len is not set!" >&2; exit 1; }

      if [ $is_zh_en -eq 1 ]; then
            tokenizer=emilia
      else
            tokenizer=espeak
            [ -z "$lang" ] && { echo "Error: lang is not set!" >&2; exit 1; }
      fi
      python3 -m zipvoice.bin.train_zipvoice \
            --world-size 4 \
            --use-fp16 1 \
            --finetune 1 \
            --base-lr 0.0001 \
            --num-iters 10000 \
            --save-every-n 1000 \
            --max-duration 500 \
            --max-len ${max_len} \
            --model-config download/zipvoice/zipvoice_base.json \
            --checkpoint download/zipvoice/model.pt \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --token-file download/zipvoice/tokens.txt \
            --dataset custom \
            --train-manifest data/fbank/custom_cuts_train.jsonl.gz \
            --dev-manifest data/fbank/custom_cuts_dev.jsonl.gz \
            --exp-dir exp/zipvoice_finetune

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Average the checkpoints for ZipVoice"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 10000 \
            --avg 2 \
            --model-name zipvoice \
            --model-config download/zipvoice/zipvoice_base.json \
            --token-file download/zipvoice/tokens.txt \
            --exp-dir exp/zipvoice_finetune
      # The generated model is exp/zipvoice_finetune/iter-10000-avg-2.pt
fi

### Inference with PyTorch models (6)

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Inference of the ZipVoice model"

      if [ $is_zh_en -eq 1 ]; then
            tokenizer=emilia
      else
            tokenizer=espeak
            [ -z "$lang" ] && { echo "Error: lang is not set!" >&2; exit 1; }
      fi

      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice \
            --checkpoint exp/zipvoice_finetune/iter-10000-avg-2.pt \
            --model-config download/zipvoice/zipvoice_base.json \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --token-file download/zipvoice/tokens.txt \
            --test-list test.tsv \
            --res-dir results/test_finetune\
            --num-step 16
fi
