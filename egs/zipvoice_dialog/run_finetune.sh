#!/bin/bash

# This script is an example of fine-tune our pre-trained ZipVoice-Dialog on your custom datasets.
# Only support English and Chinese for now.

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
# Maximum length (seconds) of the training utterance, will filter out longer utterances
max_len=60
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

      mkdir -p ${download_dir}
      hf_repo=k2-fsa/ZipVoice
      for file in model.pt tokens.txt model.json; do
            huggingface-cli download \
                  --local-dir ${download_dir} \
                  ${hf_repo} \
                  zipvoice_dialog/${file}
      done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Fine-tune the ZipVoice-Dialog model"
      python3 -m zipvoice.bin.train_zipvoice_dialog \
            --world-size 4 \
            --use-fp16 1 \
            --finetune 1 \
            --base-lr 0.0001 \
            --num-iters 10000 \
            --save-every-n 1000 \
            --max-duration 500 \
            --max-len ${max_len} \
            --checkpoint ${download_dir}/zipvoice_dialog/model.pt \
            --model-config ${download_dir}/zipvoice_dialog/model.json \
            --token-file ${download_dir}/zipvoice_dialog/tokens.txt \
            --dataset custom \
            --train-manifest data/fbank/custom_cuts_train.jsonl.gz \
            --dev-manifest data/fbank/custom_cuts_dev.jsonl.gz \
            --exp-dir exp/zipvoice_dialog_finetune
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Average the checkpoints for ZipVoice"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 10000 \
            --avg 2 \
            --model-name zipvoice_dialog \
            --model-config exp/zipvoice_dialog_finetune/model.json \
            --token-file exp/zipvoice_dialog_finetune/tokens.txt \
            --exp-dir exp/zipvoice_dialog_finetune
      # The generated model is exp/zipvoice_dialog_finetune/iter-10000-avg-2.pt
fi

### Inference with PyTorch models (6)

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Inference of the ZipVoice model"
      python3 -m zipvoice.bin.infer_zipvoice_dialog \
            --model-name zipvoice_dialog \
            --checkpoint exp/zipvoice_dialog_finetune/iter-10000-avg-2.pt \
            --model-config exp/zipvoice_dialog_finetune/model.json \
            --token-file exp/zipvoice_dialog_finetune/tokens.txt \
            --test-list test.tsv \
            --res-dir results/test_dialog_finetune
fi