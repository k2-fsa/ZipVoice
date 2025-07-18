#!/bin/bash

# This script is an example of training ZipVoice-Dialog on your custom datasets.
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
download_dir=download/

# Maximum length (seconds) of the training utterance, will filter out longer utterances
max_len=60

# We suppose you have two TSV files: "data/raw/custom_train.tsv" and 
# "data/raw/custom_dev.tsv", where "custom" is your dataset name, 
# "train"/"dev" are used for training and validation respectively.

# Each line of the TSV files should be in one of the following formats:
# (1) `{uniq_id}\t{text}\t{wav_path}` if the text corresponds to the full wav,
# (2) `{uniq_id}\t{text}\t{wav_path}\t{start_time}\t{end_time}` if text corresponds
#     to part of the wav. The start_time and end_time specify the start and end
#     times of the text within the wav, which should be in seconds.
# > Note: {uniq_id} must be unique for each line.
# > Note: {text} uses [S1] and [S2] tags to distinguish speakers, and must be begin with [S1].
# > eg: "[S1] Hello. [S2] How are you? [S1] I'm fine. [S2] What's your name?"
for subset in train dev;do
      file_path=data/raw/custom_${subset}.tsv
      [ -f "$file_path" ] || { echo "Error: expect $file_path !" >&2; exit 1; }
done


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Prepare manifests for custom dataset from tsv files"

      for subset in train dev;do
            python3 -m zipvoice.bin.prepare_dataset \
                  --tsv-path data/raw/custom_${subset}.tsv \
                  --prefix custom \
                  --subset raw_${subset} \
                  --num-jobs ${nj} \
                  --output-dir data/manifests
      done
      # The output manifest files are "data/manifests/custom_cuts_raw_train.jsonl.gz".
      # and "data/manifests/custom_cuts_raw_dev.jsonl.gz".
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Add tokens to manifests"
      for subset in train dev;do
            python3 -m zipvoice.bin.prepare_tokens \
                  --input-file data/manifests/custom_cuts_raw_${subset}.jsonl.gz \
                  --output-file data/manifests/custom_cuts_${subset}.jsonl.gz \
                  --tokenizer dialog
      done
      # The output manifest files are "data/manifests/custom_cuts_train.jsonl.gz".
      # and "data/manifests/custom_cuts_dev.jsonl.gz".
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
      echo "Stage 3: Compute Fbank for custom dataset"
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
            --world-size 4 \
            --use-fp16 1 \
            --base-lr 0.0001 \
            --num-iters 60000 \
            --max-duration 500 \
            --max-len ${max_len} \
            --checkpoint ${download_dir}/zipvoice/model.pt \
            --model-config ${download_dir}/zipvoice/model.json \
            --token-file ${download_dir}/zipvoice_dialog/tokens.txt \
            --dataset custom \
            --train-manifest data/fbank/custom_cuts_train.jsonl.gz \
            --dev-manifest data/fbank/custom_cuts_dev.jsonl.gz \
            --exp-dir exp/zipvoice_dialog_custom
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Average the checkpoints for ZipVoice"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 60000 \
            --avg 2 \
            --model-name zipvoice_dialog \
            --exp-dir exp/zipvoice_dialog_custom
      # The generated model is exp/zipvoice_dialog/iter-60000-avg-2.pt
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
      echo "Stage 6: Inference of the ZipVoice model"
      python3 -m zipvoice.bin.infer_zipvoice_dialog \
            --model-name zipvoice_dialog \
            --model-dir exp/zipvoice_dialog_custom \
            --checkpoint-name iter-60000-avg-2.pt \
            --test-list test.tsv \
            --res-dir results/test_dialog_custom
fi