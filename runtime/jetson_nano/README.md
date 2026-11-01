# Experimental Jetson Nano / TensorRT 6 runtime

This optional directory contains a fixed-shape ZipVoice-Distill decoder path for
the original Jetson Nano, CUDA 10.0 and TensorRT 6. It is independent of the
regular Python inference and `runtime/nvidia_triton` paths. It does not install
or replace a driver, CUDA, TensorRT, power mode, clock setting or system service.

The contribution includes host ONNX compatibility preparation, the decoder
builder, CUDA plugins, a reusable four-step execution bridge, FP32 cuFFT ISTFT,
component tests and a synthesis benchmark example.

## Scope

- Tested device: original Jetson Nano A02, 4 GB, Maxwell **sm53**, Ubuntu 18.04 /
  L4T R32.3.1, CUDA 10.0.326, cuDNN 7.6.3 and TensorRT 6.
- Decoder input batch is **one**, with **1024 total frames**, including the
  reference. The resident execution API evaluates **four** steps. This is a
  static deployment, not the upstream dynamic/batched inference implementation.
- The converter expects the five-input, 100-feature ZipVoice-Distill ONNX
  export: `t`, `x`, `text_condition`, `speech_condition`, `guidance_scale`.
  It validates the known 16-block attention layout rather than silently
  converting unrelated models. It promotes FP16 weights to FP32 without changing
  their values; TensorRT selects the mixed-precision execution.
- Source kernels and ideas may be reusable elsewhere, but other Maxwell GPUs
  and other TensorRT versions have **not** been validated. Build engines on the
  target device; do not copy a Nano engine to a different GPU or SDK version.
- An existing CPU ONNX text encoder and a **separately prepared FP32 TensorRT 6
  vocoder engine** are inputs to the end-to-end example. This directory does not
  download models or convert the vocoder. The vocoder must accept `mels` with
  implicit-batch binding shape `[100,target_frames]` and return frequency-major
  `[513,target_frames]` outputs `mag`, `x` (cosine), `y` (sine). Its length must
  match the prompt-dependent
  remainder of the 1024-frame decoder. Decoder tests do not certify vocoder
  conversion accuracy.
- The small lexicon frontend in `infer.py` is a Chinese benchmark helper. Use
  `--token-ids` with upstream-tokenized IDs for another frontend. It does not
  implement the full upstream English, pinyin override or long-text processing.
- One engine instance is intended for one caller. Concurrent ASR/TTS, dynamic
  lengths, dialogue models and other checkpoints are outside the tested scope.

## Build on Nano

Use the existing JetPack CUDA/TensorRT development installation. CMake 3.10 and
CUDA 10 are sufficient. No root access is required by these commands.

```sh
cd runtime/jetson_nano
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -- -j2
ctest --output-on-failure
cd ..
# Install CPU Python packages in your own environment, not the system Python:
python -m pip install -r requirements.txt
python tests/test_istft.py
```

`libzipvoice_nano.so` contains the plugin registry, engine bridge and ISTFT.
Python looks in `build/` by default; `--library` overrides the path. The plugin
type/version and serialized shape/dtype fields are retained, including when
TensorRT 6 deserializes without calling plugin configuration again.

## Prepare the decoder on a host

Perform this memory-intensive conversion on a host, not the 4 GB Nano. Start
with the FP16 ONNX decoder exported from the same ZipVoice-Distill checkpoint
as the encoder, using the upstream ONNX export flow. No weights are included
here. The source graph must have the input contract described above.

```sh
python -m pip install -r runtime/jetson_nano/requirements-host.txt
python runtime/jetson_nano/tests/test_graphs.py
python runtime/jetson_nano/tools/prepare_decoder.py \
  --input /path/to/fm_decoder_fp16.onnx --output-dir converted
```

Preparation fixes the shape, folds constant/static branches, lowers operations
unsupported by the legacy parser, adds its implicit batch axis, flattens dense
GEMMs, and reshapes last-axis Softmax to preserve the intended axis. The
ScatterND folder uses coordinate tuples, including for position encodings.
The intermediate `decoder_fixed_fp32.onnx` is the source-weight-preserving
FP32 reference. These tools intentionally reject unknown model layouts.

Copy `converted/decoder_trt6.onnx` to Nano and build the engine there:

```sh
build/build_decoder /path/to/decoder_trt6.onnx /path/to/decoder.engine
```

For a parser/graph-rewrite inspection without the long tactic build, append
`dump`. A complete Nano decoder build can take tens of minutes. CUDA plugin
tests and a successful parse are separate from a completed engine build.

## Four-step synthesis example

Use your own matching encoder, tokenizer files, prompt and vocoder engine.
This example deliberately requires an explicit `--fit-frames`: it finds a speed
which lets the CPU encoder produce exactly 1024 frames. The selected speed is
reported and reused for each steady-state run. It does not crop encoder output
or interpolate conditioning features.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python infer.py \
  --encoder /path/to/encoder.int8.onnx \
  --decoder /path/to/decoder.engine \
  --vocoder /path/to/vocos_fp32.engine \
  --tokens /path/to/tokens.txt --lexicon /path/to/lexicon.txt \
  --prompt-wav /path/to/prompt.wav --prompt-text '参考音频的文本。' \
  --text '今天测试语音合成的运行速度。' \
  --fit-frames --repeats 5 --save-mel --save-feeds --output-dir results/run1
```

Alternatively, `--token-ids ids.json` replaces the text/lexicon arguments.
Its format is `{"text":[...],"prompt":[...]}` with integer IDs for the matching
model. Prompt audio must be nonempty mono 24 kHz. To use a known speed without
searching, omit `--fit-frames` and set `--speed`; a length mismatch is an error.

The example performs one warmup and writes JSONL timing plus WAVs. The timed
region includes one CPU encoder pass, random initial state, four decoder steps,
vocoder, GPU ISTFT and the existing silence trim. Model loading, prompt feature
preparation, speed fitting and output file writing are excluded. Use a new
output directory for each process; previous measurements are never appended.

## Decoder reference and resident checks

`infer.py --save-feeds` writes real warmup inputs as `decoder_inputs.npz`.
On a host, create a reference at three seeds and four times:

```sh
python runtime/jetson_nano/tests/make_reference.py --model converted/decoder_fixed_fp32.onnx \
  --inputs decoder_inputs.npz --output reference.npz
```

On Nano:

```sh
python tests/check_decoder.py --engine /path/to/decoder.engine \
  --inputs decoder_inputs.npz --reference reference.npz --output decoder_check.json
```

The checker requires finite outputs, relative RMSE <=0.5% against this FP32
reference, and bitwise equality between host and resident four-step integration.
It exits nonzero on failure. For release validation, also compare raw mels and
sample counts, measure five runs without a profiler, and repeat in a new process.

## Measurements and implementation

The original fixed-workload deployment measured mean **RTF 0.770828** over five
runs, with **0.769426** in a separate process: approximately 3.74 s to produce
4.85 s of audio. It used an INT8 CPU encoder, the original source decoder weights,
four steps, a 6.0567 s prompt (568 frames), 456 target frames, guidance 1 and
`t_shift=0.5`. Its fitted speed was 1.0125558376312256. One seed retained the
baseline trim to 4.832 s. These are one workload's measurements, not a claim
about arbitrary text lengths or all Maxwell devices.

Maximum decoder relative RMSE against the FP32 execution reference was 0.4494%;
minimum raw-mel cosine against the prior fused3 deployment was 0.9997916.
The 7.6% latency reduction is relative to our own prior optimized Nano runtime
(RTF 0.8343), **not** an upstream PyTorch or TensorRT baseline. Earlier
power-throttled results are excluded. The benchmark retained a separately
prepared legacy GPU vocoder; its conversion is not part of this PR.

The retained optimizations are:

- Correct relative-position indexing and fused score/Softmax; exact operation
  order with fixed 256/512/1024 attention shapes.
- GLU with paired accesses and fixed-512-channel indexing, preserving the gate
  rounding before multiplication.
- Swoosh's complete FP16 input/output lookup. Hot entries use shared memory;
  other inputs use the same full table. No approximate model output is cached.
- Four-step state/conditioning residency, reusable buffers and one CUDA stream.
- FP32 cuFFT C2R plus periodic Hann, ascending overlap-add, normalization and
  centered cropping. It matches the CPU ISTFT within measured FP32 FFT error.

Engine topology changes which exceeded the numerical gate were excluded.
Experiments and model/engine binaries are not included in the contribution.
The cleaned PR library/CLI retest measured mean RTF **0.770081**, with **0.771846**
in a new process; all five raw mels were bitwise equal to the accepted prototype.
See `benchmarks/nano_1024.json` for concise measurement and validation metadata.

This is a draft contribution for feedback on legacy-backend scope and upkeep.
The clean-tree host conversion, library build, component checks and existing-
engine execution are validated separately; a complete decoder engine rebuild
after the packaging cleanup has not yet been repeated.

Feature extraction, timestamps and silence trimming in `features.py` are adapted
from the Apache-2.0 [sherpa-onnx ZipVoice example](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/zipvoice/zh-en/test_onnx.py),
retaining its copyright. This runtime's exported engines and model files are
external artifacts, not part of the contribution.
