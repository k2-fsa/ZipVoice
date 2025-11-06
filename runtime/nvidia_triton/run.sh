stage=${1:-0}
stop_stage=${2:-99}
model_name=${3:-zipvoice_distill}

echo "Start stage: $stage, Stop stage: $stop_stage"
echo "Model name: $model_name"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/workspace/ZipVoice

MODEL_DIR=models # huggingface model dir
MODEL_REPO=./model_repo_${model_name}

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
    echo "Stage 1: Download huggingface models"
    hf download k2-fsa/ZipVoice --local-dir $MODEL_DIR || exit 1
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
    echo "Stage 2: Export Zipvoice TensorRT model"
    python3 -m zipvoice.bin.tensorrt_export \
        --model-name $model_name \
        --model-dir $MODEL_DIR/$model_name \
        --checkpoint-name model.pt \
        --max-batch-size 16 \
        --trt-engine-file-name fm_decoder.fp16.plan \
        --tensorrt-model-dir $MODEL_DIR/${model_name}_trt || exit 1
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
    echo "Building triton server"
    rm -r $MODEL_REPO
    cp -r ./model_repo $MODEL_REPO
    python3 scripts/fill_template.py -i $MODEL_REPO/zipvoice/config.pbtxt model_dir:$MODEL_DIR/$model_name,model_name:$model_name,trt_engine_path:$MODEL_DIR/${model_name}_trt/fm_decoder.fp16.plan
fi

if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
    echo "Starting triton server"
    tritonserver --model-repository=$MODEL_REPO
fi

if [ "$stage" -le 5 ] && [ "$stop_stage" -ge 5 ]; then
    echo "Testing triton server"
    num_tasks=(1 2 4 8)
    split_name=wenetspeech4tts
    for num_task in ${num_tasks[@]}; do
        log_dir=./log_${model_name}_concurrent_${num_task}_${split_name}
        python3 client_grpc.py  \
                --num-tasks $num_task --huggingface-dataset yuekai/seed_tts_cosy2 \
                --split-name $split_name --log-dir $log_dir
    done
fi

if [ "$stage" -le 6 ] && [ "$stop_stage" -ge 6 ]; then
    echo "Testing http client"
    wget -nc https://raw.githubusercontent.com/SparkAudio/Spark-TTS/main/example/prompt_audio.wav -O prompt.wav
    python3 client_http.py --reference-audio prompt.wav \
        --reference-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
        --target-text "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。" \
        --output-audio "./test.wav"
fi

if [ "$stage" -le 7 ] && [ "$stop_stage" -ge 7 ]; then
    echo "Starting pytriton server"
    wget -nc https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav -O prompt_short.wav
    python3 pytriton_server.py  \
        --model_dir $MODEL_DIR/$model_name \
        --model_name $model_name \
        --trt_engine_path $MODEL_DIR/${model_name}_trt/fm_decoder.fp16.plan \
        --reference_audio_sample_rate 16000 \
        --port 8000 \
        --max_batch_size 4 \
        --use_speaker_cache \
        --prompt_audio prompt_short.wav \
        --prompt_text "希望你以后能够做得比我还好呦。"
fi

if [ "$stage" -le 8 ] && [ "$stop_stage" -ge 8 ]; then
    echo "Testing pytriton server with speaker cache"
    num_tasks=(1 2 4 8)
    split_name=wenetspeech4tts
    for num_task in ${num_tasks[@]}; do
        log_dir=./log_spk_cache_pytriton_${model_name}_concurrent_${num_task}_${split_name}
        python3 client_grpc.py  --num-tasks $num_task --huggingface-dataset yuekai/seed_tts_cosy2 --split-name $split_name --log-dir $log_dir --use-spk2info-cache True
    done
fi