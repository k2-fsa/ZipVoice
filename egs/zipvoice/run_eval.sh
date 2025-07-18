#!/bin/bash

# This script is an example of evaluate TTS models with objective metrics reported in ZipVoice paper.

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=7

download_dir=download/

# Uncomment this line to use HF mirror
# export HF_ENDPOINT=https://hf-mirror.com

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Download test sets (LibriSpeech-PC and Seed-TTS)"

      hf_repo=k2-fsa/TTS_eval_datasets
      mkdir -p ${download_dir}/
      for file in librispeech_pc_testset.tar.gz seedtts_testset.tar.gz; do
            echo "Downloading ${file}..."
            huggingface-cli download \
                  --repo-type dataset \
                  --local-dir ${download_dir}/ \
                  ${hf_repo} \
                  ${file}
            echo "Extracting ${file}..."
            tar -xzf ${download_dir}/${file} -C ${download_dir}/
      done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Download all required evaluation models"
      hf_repo=k2-fsa/TTS_eval_models
      mkdir -p ${download_dir}/tts_eval_models
      huggingface-cli download \
        --local-dir ${download_dir}/tts_eval_models \
        ${hf_repo}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Inference with the pre-trained ZipVoice model from huggingface"

      for testset in librispeech_pc seedtts_en seedtts_zh; do 

        if [ "$testset" = "librispeech_pc" ]; then
                test_tsv=${download_dir}/librispeech_pc_testset/test.tsv
        
        elif [ "$testset" = "seedtts_en" ]; then
                test_tsv=${download_dir}/seedtts_testset/en/test.tsv
        elif [ "$testset" = "seedtts_zh" ]; then
                test_tsv=${download_dir}/seedtts_testset/zh/test.tsv
        else
                echo "Error: unknown testset ${testset}" >&2
                exit 1
        fi
        echo "Inference on tetset ${testset}..."
        python3 -m zipvoice.bin.infer_zipvoice \
                --model-name zipvoice \
                --test-list ${test_tsv} \
                --res-dir results/${testset}
      done
fi



if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Evaluation on LibriSpeech-PC"
      model_path=${download_dir}/tts_eval_models
      wav_path=results/librispeech_pc
      test_tsv=${download_dir}/librispeech_pc_testset/test.tsv
      # Use LibriSpeech style transcripts for WER evaluation
      transcript_tsv=${download_dir}/librispeech_pc_testset/transcript.tsv

      python3 -m zipvoice.eval.speaker_similarity.sim \
            --wav-path ${wav_path} \
            --test-list ${test_tsv} \
            --model-dir ${model_path} 

      python3 -m zipvoice.eval.wer.hubert \
            --wav-path ${wav_path} \
            --test-list ${transcript_tsv} \
            --model-dir ${model_path} 

      python3 -m zipvoice.eval.mos.utmos \
            --wav-path ${wav_path} \
            --model-dir ${model_path} 
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Evaluation on Seed-TTS test en"
      model_path=${download_dir}/tts_eval_models
      wav_path=results/seedtts_en
      test_tsv=${download_dir}/seedtts_testset/en/test.tsv

      python3 -m zipvoice.eval.speaker_similarity.sim \
            --wav-path ${wav_path} \
            --test-list ${test_tsv} \
            --model-dir ${model_path} 

      python3 -m zipvoice.eval.wer.seedtts \
            --wav-path ${wav_path} \
            --test-list ${test_tsv} \
            --model-dir ${model_path} \
            --lang en

      python3 -m zipvoice.eval.mos.utmos \
            --wav-path ${wav_path} \
            --model-dir ${model_path} 
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Evaluation on Seed-TTS test en"
      model_path=${download_dir}/tts_eval_models
      wav_path=results/seedtts_zh
      test_tsv=${download_dir}/seedtts_testset/zh/test.tsv

      python3 -m zipvoice.eval.speaker_similarity.sim \
            --wav-path ${wav_path} \
            --test-list ${test_tsv} \
            --model-dir ${model_path} 

      python3 -m zipvoice.eval.wer.seedtts \
            --wav-path ${wav_path} \
            --test-list ${test_tsv} \
            --model-dir ${model_path} \
            --lang zh

      python3 -m zipvoice.eval.mos.utmos \
            --wav-path ${wav_path} \
            --model-dir ${model_path} 
fi