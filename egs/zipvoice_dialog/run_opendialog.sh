#!/bin/bash

# This script is an example of training ZipVoice-Dialog on OpenDialog dataset.

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

# We assume that you have downloaded the OpenDialog dataset 
# to download/OpenDialog and untarred the tar files in audio/en 
# and audio/zh so that the mp3 files are placed under these two directories.

# Download OpenDialog at https://huggingface.co/datasets/k2-fsa/OpenDialog
# or https://www.modelscope.cn/datasets/k2-fsa/OpenDialog
data_dir=download/OpenDialog
download_dir=download/

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Prepare manifests for OpenDialog dataset"

      python3 local/prepare_opendialog.py \
            --dataset-path ${data_dir} \
            --num-jobs ${nj} \
            --output-dir data/manifests
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Add tokens to manifests"
      for subset in ZH-dev ZH-train EN-dev EN-train;do
            python3 -m zipvoice.bin.prepare_tokens \
                  --input-file data/manifests/opendialog_cuts_raw_${subset}.jsonl.gz \
                  --output-file data/manifests/opendialog_cuts_${subset}.jsonl.gz \
                  --tokenizer dialog
      done
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
      echo "Stage 3: Compute Fbank for opendialog dataset"
      # You can skip this step and use `--on-the-fly-feats 1` in training stage
      for subset in ZH-dev ZH-train EN-dev EN-train;do
            python3 -m zipvoice.bin.compute_fbank \
                  --source-dir data/manifests \
                  --dest-dir data/fbank \
                  --dataset opendialog \
                  --subset ${subset} \
                  --num-jobs ${nj}
      done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Download tokens file, pretrained models"
      # Uncomment this line to use HF mirror
      # export HF_ENDPOINT=https://hf-mirror.com

      # The token file is obtained by extending some tokens 
      # on the bases of the Emilia token file.
      mkdir -p ${download_dir}
      hf_repo=k2-fsa/ZipVoice
      huggingface-cli download \
            --local-dir ${download_dir} \
            ${hf_repo} \
            zipvoice_dialog/tokens.txt
      
      # Pre-trained ZipVoice model is required as 
      # the initialization model.
      for file in model.pt tokens.txt model.json; do
            huggingface-cli download \
                  --local-dir ${download_dir} \
                  ${hf_repo} \
                  zipvoice/${file}
      done
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Train the ZipVoice-Dialog model"
      python3 -m zipvoice.bin.train_zipvoice_dialog \
            --world-size 8 \
            --use-fp16 1 \
            --base-lr 0.0001 \
            --max-duration 500 \
            --checkpoint ${download_dir}/zipvoice/model.pt \
            --model-config ${download_dir}/zipvoice/model.json \
            --token-file ${download_dir}/zipvoice_dialog/tokens.txt \
            --dataset opendialog \
            --manifest-dir data/fbank \
            --exp-dir exp/zipvoice_dialog_opendialog
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Average the checkpoints for ZipVoice"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 60000 \
            --avg 2 \
            --model-name zipvoice_dialog \
            --exp-dir exp/zipvoice_dialog_opendialog
      # The generated model is exp/zipvoice_dialog_opendialog/iter-60000-avg-2.pt
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
      echo "Stage 7: Inference of the ZipVoice model"

      python3 -m zipvoice.bin.infer_zipvoice_dialog \
            --model-name zipvoice_dialog \
            --model-dir exp/zipvoice_dialog_opendialog \
            --checkpoint-name iter-60000-avg-2.pt \
            --test-list test.tsv \
            --res-dir results/test_dialog
fi