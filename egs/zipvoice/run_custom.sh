#!/bin/bash

# This script is an example of training ZipVoice on your custom datasets from scratch.

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

# You can set `train_hours` and `max_len` according to statistics from
# the command `lhotse cut describe data/fbank/custom_cuts_train.jsonl.gz`.
# Set `train_hours` to "Total speech duration", and set `max_len` to 99% duration.

# Number of hours in training set, will affect the learning rate schedule
train_hours=500
# Maximum length (seconds) of the training utterance, will filter out longer utterances
max_len=20

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
            python3 -m zipvoice.bin.prepare_dataset \
                  --tsv-path data/raw/custom_${subset}.tsv \
                  --prefix custom \
                  --subset ${subset} \
                  --num-jobs ${nj} \
                  --output-dir data/manifests
      done
      # The output manifest files are "data/manifests/custom_cuts_train.jsonl.gz".
      # and "data/manifests/custom_cuts_dev.jsonl.gz".

      # We did not add tokens to the manifests, as on-the-fly tokenization
      # with the simple tokenizer used in this example is not slow.
      # If you change to a complex tokenizer, e.g., with g2p and heavy text normalization,
      # you may need to add tokens to the manifests to speed up the training.
      # Refer to the fine-tuning example for adding tokens to the manifests.
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
      echo "Stage 3: Prepare tokens file for custom dataset"
      # In this example, we use the simplest tokenizer that 
      #     treat every character as a token.
      python3 ./local/prepare_token_file_char.py \
            --manifest data/manifests/custom_cuts_train.jsonl.gz \
            --tokens data/tokens_custom.txt
fi


### Training (4 - 5)

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Train the ZipVoice model"

      [ -z "$train_hours" ] && { echo "Error: train_hours is not set!" >&2; exit 1; }
      [ -z "$max_len" ] && { echo "Error: max_len is not set!" >&2; exit 1; }

      # lr-hours will be set according to the `train_hours`,
      # i.e., lr_hours = 1000 * (train_hours ** 0.3).
      lr_hours=$(python3 -c "print(round(1000 * ($train_hours ** 0.3)))" )
      python3 -m zipvoice.bin.train_zipvoice \
            --world-size 4 \
            --use-fp16 1 \
            --num-iters 60000 \
            --max-duration 500 \
            --lr-hours ${lr_hours} \
            --max-len ${max_len} \
            --model-config conf/zipvoice_base.json \
            --tokenizer simple \
            --token-file data/tokens_custom.txt \
            --dataset custom \
            --train-manifest data/fbank/custom_cuts_train.jsonl.gz \
            --dev-manifest data/fbank/custom_cuts_dev.jsonl.gz \
            --exp-dir exp/zipvoice_custom
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Average the checkpoints for ZipVoice"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 60000 \
            --avg 2 \
            --model-name zipvoice \
            --exp-dir exp/zipvoice_custom
      # The generated model is exp/zipvoice_custom/iter-60000-avg-2.pt
fi

### Inference with PyTorch models (6)

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Inference of the ZipVoice model"
      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice \
            --model-dir exp/zipvoice_custom \
            --checkpoint-name iter-60000-avg-2.pt \
            --tokenizer simple \
            --test-list test.tsv \
            --res-dir results/test_custom
fi
