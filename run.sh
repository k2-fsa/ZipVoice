start_stage=$1
stop_stage=$2

if [ $start_stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Download prompt audio"
    pip install -r requirements.txt
    huggingface-cli download k2-fsa/ZipVoice --local-dir models
fi

if [ $start_stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Export ONNX model"
    python3 -m zipvoice.bin.onnx_export \
        --model-name zipvoice_distill \
        --model-dir models/zipvoice_distill \
        --checkpoint-name model.pt \
        --onnx-model-dir models/zipvoice_distill_onnx_trt || exit 1

    polygraphy surgeon sanitize models/zipvoice_distill_onnx_trt/fm_decoder.onnx --fold-constant -o models/zipvoice_distill_onnx_trt/fm_decoder.simplified.onnx
fi


if [ $start_stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Run inference"
    wget -nc https://raw.githubusercontent.com/SparkAudio/Spark-TTS/main/example/prompt_audio.wav -O prompt.wav
    python3 -m zipvoice.bin.infer_zipvoice \
        --model-name zipvoice_distill \
        --prompt-wav prompt.wav \
        --prompt-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
        --text "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。" \
        --res-wav-path result_trt.wav --enable-trt True
fi

if [ $start_stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Run inference"
    python3 -m zipvoice.bin.infer_zipvoice \
        --model-name zipvoice_distill \
        --enable-trt True \
        --huggingface-dataset-split wenetspeech4tts \
        --batch-size 4 --enable-warmup True --res-dir results_4_steps_new --num-step 4
fi
