#!/bin/bash

# This is an example script for training ZipVoice on Emilia dataset.

# This script covers data preparation, ZipVoice trainnig, 
#     ZipVoice-Distill training, onnx export, and 
#     inference with all PyTorch and ONNX models.


# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=12

#### Prepare datasets (1)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Data Preparation for Emilia dataset"
      bash local/prepare_emilia.sh
fi

### Training ZipVoice (2 - 3)

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Train the ZipVoice model"
      python3 -m zipvoice.bin.train_zipvoice \
            --world-size 8 \
            --use-fp16 1 \
            --num-epochs 11 \
            --max-duration 500 \
            --lr-hours 30000 \
            --model-config conf/zipvoice_base.json \
            --tokenizer emilia \
            --token-file data/tokens_emilia.txt \
            --dataset emilia \
            --manifest-dir data/fbank \
            --exp-dir exp/zipvoice
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Average the checkpoints for ZipVoice"
      python3 -m zipvoice.bin.generate_averaged_model \
            --epoch 11 \
            --avg 4 \
            --model-name zipvoice \
            --exp-dir exp/zipvoice
      # The generated model is exp/zipvoice/epoch-11-avg-4.pt
fi

#### (Optional) Training ZipVoice-Distill model (4 - 6)

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Train the ZipVoice-Distill model (first stage)"
      python3 -m zipvoice.bin.train_zipvoice_distill \
            --world-size 8 \
            --use-fp16 1 \
            --num-iters 60000 \
            --max-duration 500 \
            --base-lr 0.0005 \
            --tokenizer emilia \
            --token-file data/tokens_emilia.txt \
            --dataset emilia \
            --manifest-dir data/fbank \
            --teacher-model zipvoice/exp_zipvoice/epoch-11-avg-4.pt \
            --distill-stage first \
            --exp-dir exp/zipvoice_distill_1stage
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Average the checkpoints for ZipVoice-Distill (first stage)"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 60000 \
            --avg 7 \
            --model-name zipvoice_distill \
            --exp-dir exp/zipvoice_distill_1stage
      # The generated model is exp/zipvoice_distill_1stage/iter-60000-avg-7.pt
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Train the ZipVoice-Distill model (second stage)"

      python3 -m zipvoice.bin.train_zipvoice_distill \
            --world-size 8 \
            --use-fp16 1 \
            --num-iters 2000 \
            --save-every-n 1000 \
            --max-duration 500 \
            --base-lr 0.0001 \
            --model-config conf/zipvoice_base.json \
            --tokenizer emilia \
            --token-file data/tokens_emilia.txt \
            --dataset emilia \
            --manifest-dir data/fbank \
            --teacher-model exp/zipvoice_distill_1stage/iter-60000-avg-7.pt \
            --distill-stage second \
            --exp-dir exp/zipvoice_distill
fi

### Export ONNX model (7 - 8)

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
      echo "Stage 7: Export ZipVoice ONNX model"
      python3 -m zipvoice.bin.onnx_export \
            --model-name zipvoice \
            --model-dir exp/zipvoice/ \
            --checkpoint-name epoch-11-avg-4.pt \
            --onnx-model-dir exp/zipvoice/
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
      echo "Stage 8: Export ZipVoice-Distill ONNX model"
      python3 -m zipvoice.bin.onnx_export \
            --model-name zipvoice_distill \
            --model-dir exp/zipvoice_distill/ \
            --checkpoint-name checkpoint-2000.pt \
            --onnx-model-dir exp/zipvoice_distill/
fi


### Inference with PyTorch and ONNX models (9 - 12)

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
      echo "Stage 9: Inference of the ZipVoice model"
      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice \
            --model-dir exp/zipvoice/ \
            --checkpoint-name epoch-11-avg-4.pt \
            --tokenizer emilia \
            --test-list test.tsv \
            --res-dir results/test \
            --num-step 16 \
            --guidance-scale 1 \
            --raw-evaluation True
fi


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
      echo "Stage 10: Inference of the ZipVoice-Distill model"
      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice_distill \
            --model-dir exp/zipvoice_distill/ \
            --checkpoint-name checkpoint-2000.pt \
            --tokenizer emilia \
            --test-list test.tsv \
            --res-dir results/test_distill \
            --num-step 8 \
            --guidance-scale 3 \
            --raw-evaluation True
fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
      echo "Stage 11: Inference with ZipVoice ONNX model"
      python3 -m zipvoice.bin.infer_zipvoice_onnx \
            --model-name zipvoice \
            --onnx-int8 False \
            --model-dir exp/zipvoice \
            --tokenizer emilia \
            --test-list test.tsv \
            --res-dir results/test_onnx
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
      echo "Stage 12: Inference with ZipVoic-Distill ONNX model"
      python3 -m zipvoice.bin.infer_zipvoice_onnx \
            --model-name zipvoice_distill \
            --onnx-int8 False \
            --model-dir exp/zipvoice_distill \
            --tokenizer emilia \
            --test-list test.tsv \
            --res-dir results/test_distill_onnx
fi