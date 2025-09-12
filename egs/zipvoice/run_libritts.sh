#!/bin/bash

# This is an example script for training ZipVoice on LibriTTS dataset.

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=9

#### Prepare datasets (1)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Data Preparation for LibriTTS dataset"
      bash local/prepare_libritts.sh
fi

### Training ZipVoice (2 - 3)

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Train the ZipVoice model"
      python3 -m zipvoice.bin.train_zipvoice \
            --world-size 8 \
            --use-fp16 0 \
            --num-epochs 60 \
            --max-duration 250 \
            --lr-epochs 10 \
            --max-len 20 \
            --valid-by-epoch 1 \
            --model-config conf/zipvoice_base.json \
            --tokenizer libritts \
            --token-file data/tokens_libritts.txt \
            --dataset libritts \
            --manifest-dir data/fbank \
            --exp-dir exp/zipvoice_libritts
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Average the checkpoints for ZipVoice"
      python3 -m zipvoice.bin.generate_averaged_model \
            --epoch 60 \
            --avg 10 \
            --model-name zipvoice \
            --exp-dir exp/zipvoice_libritts
      # The generated model is exp/zipvoice_libritts/epoch-60-avg-10.pt
fi

#### (Optional) Training ZipVoice-Distill model (4 - 7)

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Train the ZipVoice-Distill model (first stage)"
      python3 -m zipvoice.bin.train_zipvoice_distill \
            --world-size 8 \
            --use-fp16 0 \
            --num-epochs 6 \
            --max-duration 250 \
            --base-lr 0.001 \
            --max-len 20 \
            --valid-by-epoch 1 \
            --model-config conf/zipvoice_base.json \
            --tokenizer libritts \
            --token-file data/tokens_libritts.txt \
            --dataset "libritts" \
            --manifest-dir "data/fbank" \
            --teacher-model exp/zipvoice_libritts/epoch-60-avg-10.pt \
            --distill-stage "first" \
            --exp-dir exp/zipvoice_distill_1stage_libritts
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Average the checkpoints for ZipVoice-Distill (first stage)"
      python3 -m zipvoice.bin.generate_averaged_model \
            --epoch 6 \
            --avg 3 \
            --model-name zipvoice_distill \
            --exp-dir exp/zipvoice_distill_1stage_libritts
      # The generated model is exp/zipvoice_distill_1stage_libritts/epoch-6-avg-3.pt
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Train the ZipVoice-Distill model (second stage)"

      python3 -m zipvoice.bin.train_zipvoice_distill \
            --world-size 8 \
            --use-fp16 1 \
            --num-epochs 6 \
            --max-duration 250 \
            --base-lr 0.001 \
            --max-len 20 \
            --valid-by-epoch 1 \
            --model-config conf/zipvoice_base.json \
            --tokenizer libritts \
            --token-file data/tokens_libritts.txt \
            --dataset libritts \
            --manifest-dir data/fbank \
            --teacher-model exp/zipvoice_distill_1stage_libritts/epoch-6-avg-3.pt \
            --distill-stage second \
            --exp-dir exp/zipvoice_distill_libritts
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
      echo "Stage 7: Average the checkpoints for ZipVoice-Distill (second stage)"
      python3 -m zipvoice.bin.generate_averaged_model \
            --epoch 6 \
            --avg 3 \
            --model-name zipvoice_distill \
            --exp-dir exp/zipvoice_distill_libritts
      # The generated model is exp/zipvoice_distill_libritts/epoch-6-avg-3.pt
fi

### Inference with PyTorch models (8 - 9)

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
      echo "Stage 8: Inference of the ZipVoice model"
      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice \
            --model-dir exp/zipvoice_libritts \
            --checkpoint-name epoch-60-avg-10.pt \
            --tokenizer libritts \
            --test-list test.tsv \
            --res-dir results/test_libritts \
            --num-step 8 \
            --guidance-scale 1 \
            --t-shift 0.7 \
            --raw-evaluation True
fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
      echo "Stage 9: Inference of the ZipVoice-Distill model"
      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice_distill \
            --model-dir exp/zipvoice_distill_libritts \
            --checkpoint-name epoch-6-avg-3.pt \
            --tokenizer libritts \
            --test-list test.tsv \
            --res-dir results/test_distill_libritts \
            --num-step 4 \
            --guidance-scale 3 \
            --t-shift 0.7 \
            --raw-evaluation True
fi
