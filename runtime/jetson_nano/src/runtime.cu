// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
// See the repository LICENSE for the full license text.

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string last_error;
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING)
      std::fprintf(stderr, "TensorRT: %s\n", message);
    if (severity <= Severity::kERROR) last_error = message;
  }
};
static Logger logger;
static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static size_t type_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kINT8:
      return 1;
    default:
      throw std::runtime_error("Unsupported binding dtype");
  }
}
struct Engine {
  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, end = nullptr;
  std::vector<void*> buffers;
  std::vector<size_t> bytes, elements;
  float last_ms = 0;
  ~Engine() {
    if (stream) cudaStreamSynchronize(stream);
    for (void* b : buffers)
      if (b) cudaFree(b);
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
    if (start) cudaEventDestroy(start);
    if (end) cudaEventDestroy(end);
    if (stream) cudaStreamDestroy(stream);
  }
};
extern "C" {
const char* nano_trt_error() { return last_error.c_str(); }
void* nano_trt_create(const char* filename) {
  Engine* e = new Engine;
  try {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open engine");
    auto length = in.tellg();
    if (length <= 0) throw std::runtime_error("Empty engine");
    std::vector<char> blob(static_cast<size_t>(length));
    in.seekg(0);
    in.read(blob.data(), blob.size());
    if (!in) throw std::runtime_error("Incomplete engine read");
    e->runtime = nvinfer1::createInferRuntime(logger);
    if (!e->runtime) throw std::runtime_error("Cannot create runtime");
    e->engine =
        e->runtime->deserializeCudaEngine(blob.data(), blob.size(), nullptr);
    if (!e->engine)
      throw std::runtime_error("Cannot deserialize engine: " + last_error);
    e->context = e->engine->createExecutionContext();
    if (!e->context) throw std::runtime_error("Cannot create context");
    check(cudaStreamCreate(&e->stream));
    check(cudaEventCreate(&e->start));
    check(cudaEventCreate(&e->end));
    int count = e->engine->getNbBindings();
    e->buffers.resize(count, nullptr);
    e->bytes.resize(count);
    e->elements.resize(count);
    for (int i = 0; i < count; i++) {
      auto dims = e->engine->getBindingDimensions(i);
      size_t size = 1;
      for (int j = 0; j < dims.nbDims; j++) {
        if (dims.d[j] <= 0)
          throw std::runtime_error("Dynamic or empty binding");
        size *= dims.d[j];
      }
      e->elements[i] = size;
      e->bytes[i] = size * type_size(e->engine->getBindingDataType(i));
      check(cudaMalloc(&e->buffers[i], e->bytes[i]));
    }
    return e;
  } catch (const std::exception& ex) {
    last_error = ex.what();
    delete e;
    return nullptr;
  }
}
void nano_trt_destroy(void* h) { delete static_cast<Engine*>(h); }
int nano_trt_count(void* h) {
  return static_cast<Engine*>(h)->engine->getNbBindings();
}
const char* nano_trt_name(void* h, int i) {
  return static_cast<Engine*>(h)->engine->getBindingName(i);
}
int nano_trt_is_input(void* h, int i) {
  return static_cast<Engine*>(h)->engine->bindingIsInput(i);
}
int nano_trt_dtype(void* h, int i) {
  return static_cast<int>(
      static_cast<Engine*>(h)->engine->getBindingDataType(i));
}
size_t nano_trt_elements(void* h, int i) {
  return static_cast<Engine*>(h)->elements.at(i);
}
int nano_trt_rank(void* h, int i) {
  return static_cast<Engine*>(h)->engine->getBindingDimensions(i).nbDims;
}
int nano_trt_dim(void* h, int i, int j) {
  return static_cast<Engine*>(h)->engine->getBindingDimensions(i).d[j];
}
float nano_trt_last_ms(void* h) { return static_cast<Engine*>(h)->last_ms; }
int nano_trt_run(void* h, void** host) {
  Engine* e = static_cast<Engine*>(h);
  try {
    check(cudaEventRecord(e->start, e->stream));
    for (int i = 0; i < e->engine->getNbBindings(); i++)
      if (e->engine->bindingIsInput(i))
        check(cudaMemcpyAsync(e->buffers[i], host[i], e->bytes[i],
                              cudaMemcpyHostToDevice, e->stream));
    if (!e->context->enqueue(1, e->buffers.data(), e->stream, nullptr))
      throw std::runtime_error("TensorRT enqueue failed: " + last_error);
    for (int i = 0; i < e->engine->getNbBindings(); i++)
      if (!e->engine->bindingIsInput(i))
        check(cudaMemcpyAsync(host[i], e->buffers[i], e->bytes[i],
                              cudaMemcpyDeviceToHost, e->stream));
    check(cudaEventRecord(e->end, e->stream));
    check(cudaStreamSynchronize(e->stream));
    check(cudaEventElapsedTime(&e->last_ms, e->start, e->end));
    return 0;
  } catch (const std::exception& ex) {
    last_error = ex.what();
    return -1;
  }
}
}

__global__ void nano_integrate(float* x, const float* v, float dt, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __fadd_rn(x[i], __fmul_rn(v[i], dt));
}
extern "C" int nano_trt_run4(void* h, void** host, const float* ts,
                             const float* delta, float* result) {
  Engine* e = static_cast<Engine*>(h);
  try {
    int ix = e->engine->getBindingIndex("x"),
        iv = e->engine->getBindingIndex("v"),
        it = e->engine->getBindingIndex("t");
    if (ix < 0 || iv < 0 || it < 0)
      throw std::runtime_error("missing state bindings");
    if (e->elements[ix] != e->elements[iv] || e->elements[it] != 1)
      throw std::runtime_error("run4 requires matching state/velocity and scalar time");
    for (int j : {ix, iv, it})
      if (e->engine->getBindingDataType(j) != nvinfer1::DataType::kFLOAT)
        throw std::runtime_error("resident expects FP32 state");
    check(cudaEventRecord(e->start, e->stream));
    for (int j = 0; j < e->engine->getNbBindings(); j++)
      if (e->engine->bindingIsInput(j) && j != it)
        check(cudaMemcpyAsync(e->buffers[j], host[j], e->bytes[j],
                              cudaMemcpyHostToDevice, e->stream));
    for (int step = 0; step < 4; step++) {
      check(cudaMemcpyAsync(e->buffers[it], ts + step, sizeof(float),
                            cudaMemcpyHostToDevice, e->stream));
      if (!e->context->enqueue(1, e->buffers.data(), e->stream, nullptr))
        throw std::runtime_error("resident enqueue failed");
      nano_integrate<<<(e->elements[ix] + 255) / 256, 256, 0, e->stream>>>(
          (float*)e->buffers[ix], (const float*)e->buffers[iv], delta[step],
          e->elements[ix]);
      check(cudaPeekAtLastError());
    }
    check(cudaMemcpyAsync(result, e->buffers[ix], e->bytes[ix],
                          cudaMemcpyDeviceToHost, e->stream));
    check(cudaEventRecord(e->end, e->stream));
    check(cudaStreamSynchronize(e->stream));
    check(cudaEventElapsedTime(&e->last_ms, e->start, e->end));
    return 0;
  } catch (const std::exception& ex) {
    last_error = ex.what();
    return -1;
  }
}
