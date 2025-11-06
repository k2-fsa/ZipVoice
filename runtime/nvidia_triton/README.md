# ZipVoice Triton Inference Server

This repository provides optimized inference deployment for ZipVoice text-to-speech models using NVIDIA Triton Inference Server and PyTriton, with TensorRT acceleration for production environments.

## Table of Contents

- [Quick Start](#quick-start)
  - [Option 1: Docker Compose (Recommended)](#option-1-docker-compose-recommended)
  - [Option 2: Manual Docker Setup](#option-2-manual-docker-setup)
- [Server Deployment](#server-deployment)
  - [Understanding run.sh Stages](#understanding-runsh-stages)
  - [Triton Server Setup](#triton-server-setup)
  - [PyTriton Server Setup](#pytriton-server-setup)
- [Client Testing](#client-testing)
  - [HTTP Client](#http-client)
  - [gRPC Client](#grpc-client)
- [Performance](#performance)
  - [Benchmarking](#benchmarking)
  - [Benchmark Results](#benchmark-results)
- [Advanced Features](#advanced-features)
  - [OpenAI-Compatible API](#openai-compatible-api)
  - [Speaker Cache](#speaker-cache)
- [Acknowledgements](#acknowledgements)

## Quick Start

### Option 1: Docker Compose (Recommended)

Launch the service directly using Docker Compose:

```sh
# For standard ZipVoice model
MODEL=zipvoice docker compose up

# For distilled ZipVoice model (faster inference)
MODEL=zipvoice_distill docker compose up
```

### Option 2: Manual Docker Setup

Build and run the Docker container manually:

```sh
# Build the Docker image
docker build . -f Dockerfile.server -t soar97/triton-zipvoice:24.12

# Create and run Docker container
your_mount_dir=/your/host/path:/your/container/path
docker run -it --name "zipvoice-server" --gpus all --net host \
    -v $your_mount_dir --shm-size=2g soar97/triton-zipvoice:24.12
```

## Server Deployment

### Understanding run.sh Stages

The `run.sh` script automates the entire deployment workflow through numbered stages. Run specific stages with:

```sh
bash run.sh <start_stage> <stop_stage> [model_name]
```

- `<start_stage>`: Starting stage number (1-8)
- `<stop_stage>`: Ending stage number (1-8)  
- `[model_name]`: Optional model name (`zipvoice` or `zipvoice_distill`, default: `zipvoice_distill`)

**Available Stages:**

- **Stage 1**: Downloads ZipVoice models from HuggingFace
- **Stage 2**: Exports models to TensorRT format and builds optimized engines
- **Stage 3**: Creates Triton model repository and configuration files
- **Stage 4**: Launches Triton Inference Server
- **Stage 5**: Runs gRPC benchmark tests with multiple concurrency levels
- **Stage 6**: Tests HTTP client with sample audio
- **Stage 7**: Launches PyTriton server with speaker caching
- **Stage 8**: Tests PyTriton server with speaker cache benchmarks

### Triton Server Setup

Build TensorRT engines and launch the Triton server:

```sh
# Complete setup and launch (stages 1-4)
bash run.sh 1 4 zipvoice_distill
```

> [!NOTE]
> To modify the default NFE (Neural Function Evaluation) steps, edit `model_repo/zipvoice/1/model.py` manually.

### PyTriton Server Setup

Launch the PyTriton server with speaker caching for improved performance:

```sh
# Launch PyTriton server (stage 7)
bash run.sh 7 7 zipvoice_distill
```
> [!NOTE]
> To use the PyTriton Server, you don't have to use the Docker environment. You can install it manually with `pip install nvidia-pytriton`.


## Client Testing

### HTTP Client

Test the server with a simple HTTP client:

```sh
python3 client_http.py --reference-audio prompt.wav \
    --reference-text "Your reference text here" \
    --target-text "Text to synthesize" \
    --output-audio "./output.wav"
```

### gRPC Client

Run performance benchmarks using the gRPC client:

```sh
# Single task benchmark
python3 client_grpc.py --num-tasks 1 --huggingface-dataset yuekai/seed_tts_cosy2 \
    --split-name wenetspeech4tts

# Multi-task benchmark
num_task=8
python3 client_grpc.py --num-tasks $num_task --huggingface-dataset yuekai/seed_tts_cosy2 \
    --split-name wenetspeech4tts
```

## Performance

### Benchmarking

Run automated benchmarks across multiple concurrency levels:

```sh
# Benchmark Triton server (stage 5)
bash run.sh 5 5 zipvoice_distill

# Benchmark PyTriton server with speaker cache (stage 8)
bash run.sh 8 8 zipvoice_distill
```

### Benchmark Results

Performance metrics on a single NVIDIA L20 GPU using 26 different prompt-text pairs with ZipVoice Distill (4 NFE steps):

| Concurrency | Processing Time (s) | P50 Latency (ms) | Avg Latency (ms) |
|-------------|---------------------|------------------|------------------|
| 1           | 3.011              | 98.73           | 103.34          |
| 1   (with 3s prompt speaker cache)        | 2.652              |     88.78       |   88.34        |
| 2           | 2.261              | 158.71          | 159.49          |
| 2   (with 3s prompt speaker cache)        |    1.729          |      116.53      |    119.74      |
| 4           | 1.872              | 272.16          | 261.75          |
| 4   (with 3s prompt speaker cache)        |    1.330           |   184.19         |     179.32      |
| 8           | 1.710              | 468.29          | 470.20          |
| 8   (with 3s prompt speaker cache)        |    1.220           |       300.48     |    306.35      |

## Advanced Features

### OpenAI-Compatible API

Deploy an OpenAI-compatible TTS API service:

```sh
# Clone the OpenAI bridge repository
git clone https://github.com/yuekaizhang/Triton-OpenAI-Speech.git
cd Triton-OpenAI-Speech
pip install -r requirements.txt

# Start the FastAPI bridge (after Triton service is running)
python3 tts_server.py --url http://localhost:8000 \
    --ref_audios_dir ./ref_audios/ \
    --port 10086 \
    --default_sample_rate 24000
```

### Speaker Cache

The PyTriton server supports speaker caching to improve performance for repeated synthesis with the same reference audio:

- Enabled with `--use_speaker_cache` flag
- Reduces latency by using short-duration prompt audio (e.g., 3 seconds)

## Acknowledgements

This work originates from the NVIDIA CISI project. For additional multimodal AI resources, visit [mair-hub](https://github.com/nvidia-china-sae/mair-hub).
