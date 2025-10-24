

# wget https://raw.githubusercontent.com/SparkAudio/Spark-TTS/main/example/prompt_audio.wav -O prompt.wav

# polygraphy surgeon sanitize models/zipvoice_distill_onnx_trt/fm_decoder.onnx --fold-constant -o models/zipvoice_distill_onnx_trt/fm_decoder.simplified.onnx

# python3 -m zipvoice.bin.onnx_export \
#     --model-name zipvoice_distill \
#     --model-dir models/zipvoice_distill \
#     --checkpoint-name model.pt \
#     --onnx-model-dir models/zipvoice_distill_onnx_trt || exit 1

# polygraphy surgeon sanitize models/zipvoice_distill_onnx_trt/fm_decoder.onnx --fold-constant -o models/zipvoice_distill_onnx_trt/fm_decoder.simplified.onnx

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice_distill \
    --prompt-wav prompt.wav \
    --prompt-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
    --text "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。" \
    --res-wav-path result_trt.wav --enable-trt True


