# ZipVoice Recipe

This recipe contains the following examples:

- Training ZipVoice on Emilia from scratch, see [run_emilia.sh](run_emilia.sh)
- Training ZipVoice on LibriTTS from scratch, see [run_libritts.sh](run_libritts.sh).
- Training ZipVoice on custom datasets (any language) from scratch, see [run_custom.sh](run_custom.sh).
- Fine-tuning pre-trained ZipVoice on custom datasets (any language), see [run_finetune.sh](run_finetune.sh).
- Evaluate TTS models with objective metrics reported in ZipVoice paper, see [run_eval.sh](run_eval.sh).

> **NOTE:**  [run_emilia.sh](run_emilia.sh) is the most complete example, which covers: data preparation, ZipVoice trainnig, ZipVoice-Distill training, onnx export, and inference with all PyTorch and ONNX models.

>  **NOTE:** For evaluation, first install packages from [../../requirements_eval.txt](../../requirements_eval.txt)
> 
> `pip install -r ../../requirements_eval.txt`
