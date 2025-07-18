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

### Prepare the training data (1 - 3)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Prepare manifests for OpenDialog dataset"

      python3 local/prepare_opendialog.py \
            --dataset-path ${data_dir} \
            --num-jobs ${nj} \
            --output-dir data/manifests
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
      echo "Stage 2: Compute Fbank for custom dataset"
      # You can skip this step and use `--on-the-fly-feats 1` in training stage
      for lang in EN ZH;do
      for subset in train dev; do
            python3 -m zipvoice.bin.compute_fbank \
                  --source-dir data/manifests \
                  --dest-dir data/fbank \
                  --dataset opendialog \
                  --subset "${lang}-${subset}" \
                  --num-jobs ${nj}
      done
      done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Download tokens file, pretrained models"
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


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Train the ZipVoice-Dialog model"
      python3 -m zipvoice.bin.train_zipvoice_dialog \
            --world-size 8 \
            --use-fp16 1 \
            --base-lr 0.0001 \
            --max-duration 500 \
            --checkpoint download/zipvoice/model.pt \
            --model-config download/zipvoice/model.json \
            --token-file download/zipvoice_dialog/tokens.txt \
            --dataset opendialog \
            --manifest-dir data/fbank \
            --exp-dir exp/zipvoice_dialog_opendialog
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Average the checkpoints for ZipVoice"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 60000 \
            --avg 2 \
            --model-name zipvoice_dialog \
            --model-config exp/zipvoice_dialog_opendialog/model.json \
            --token-file exp/zipvoice_dialog_opendialog/tokens.txt \
            --exp-dir exp/zipvoice_dialog_opendialog
      # The generated model is exp/zipvoice_dialog_opendialog/iter-60000-avg-2.pt
fi

### Inference with PyTorch models (6)

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Inference of the ZipVoice model"

      python3 -m zipvoice.bin.infer_zipvoice_dialog \
            --model-name zipvoice_dialog \
            --checkpoint exp/zipvoice_dialog_opendialog/iter-60000-avg-2.pt \
            --model-config exp/zipvoice_dialog_opendialog/model.json \
            --token-file exp/zipvoice_dialog_opendialog/tokens.txt \
            --test-list test.tsv \
            --res-dir results/test_dialog
fi