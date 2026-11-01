// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
// See the repository LICENSE for the full license text.

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <vector>
using namespace nvinfer1;

template <class T>
__global__ void row_softmax(const T* in, T* out, int K) {
  int r = blockIdx.x;
  const T* x = in + (long)r * K;
  T* y = out + (long)r * K;
  extern __shared__ float sh[];
  float* smax = sh;
  float* ssum = sh + blockDim.x;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    float v = (float)x[i];
    m = fmaxf(m, v);
  }
  smax[threadIdx.x] = m;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (threadIdx.x < off)
      smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + off]);
    __syncthreads();
  }
  m = smax[0];
  float s = 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) s += expf((float)x[i] - m);
  ssum[threadIdx.x] = s;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (threadIdx.x < off) ssum[threadIdx.x] += ssum[threadIdx.x + off];
    __syncthreads();
  }
  float inv = 1.f / ssum[0];
  for (int i = threadIdx.x; i < K; i += blockDim.x)
    y[i] = (T)(expf((float)x[i] - m) * inv);
}
class RowSoftmax : public IPluginV2 {
  int rows = 0, K = 0;
  DataType dtype = DataType::kFLOAT;
  std::string ns;

 public:
  void restore(int r, int k, int dt) {
    rows = r;
    K = k;
    dtype = (DataType)dt;
  }
  const char* getPluginType() const noexcept override {
    return "NanoRowSoftmax";
  }
  const char* getPluginVersion() const noexcept override { return "1"; }
  int getNbOutputs() const noexcept override { return 1; }
  Dims getOutputDimensions(int, Dims const* in, int) noexcept override {
    return in[0];
  }
  bool supportsFormat(DataType t, PluginFormat f) const noexcept override {
    return (t == DataType::kFLOAT || t == DataType::kHALF) &&
           f == PluginFormat::kNCHW;
  }
  void configureWithFormat(const Dims* in, int nb, const Dims*, int, DataType t,
                           PluginFormat, int) noexcept override {
    dtype = t;
    if (nb < 1 || in[0].nbDims < 1) return;
    K = in[0].d[in[0].nbDims - 1];
    long r = 1;
    for (int i = 0; i < in[0].nbDims - 1; i++) r *= in[0].d[i];
    rows = (int)r;
  }
  int initialize() noexcept override { return K > 0 && rows > 0 ? 0 : -1; }
  void terminate() noexcept override {}
  size_t getWorkspaceSize(int) const noexcept override { return 0; }
  int enqueue(int batch, const void* const* in, void** out, void*,
              cudaStream_t st) noexcept override {
    int totalRows = batch * rows;
    int threads = 256;
    size_t shmem = 2 * threads * sizeof(float);
    if (dtype == DataType::kHALF)
      row_softmax<<<totalRows, threads, shmem, st>>>((const __half*)in[0],
                                                     (__half*)out[0], K);
    else
      row_softmax<<<totalRows, threads, shmem, st>>>((const float*)in[0],
                                                     (float*)out[0], K);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
  }
  size_t getSerializationSize() const noexcept override {
    return 3 * sizeof(int);
  }
  void serialize(void* d) const noexcept override {
    int p[3] = {rows, K, (int)dtype};
    std::memcpy(d, p, sizeof(p));
  }
  void destroy() noexcept override { delete this; }
  IPluginV2* clone() const noexcept override {
    auto* p = new RowSoftmax();
    p->rows = rows;
    p->K = K;
    p->dtype = dtype;
    p->ns = ns;
    return p;
  }
  void setPluginNamespace(const char* n) noexcept override { ns = n ? n : ""; }
  const char* getPluginNamespace() const noexcept override {
    return ns.c_str();
  }
};
class SCreator : public IPluginCreator {
  std::string ns;
  PluginFieldCollection f{};

 public:
  const char* getPluginName() const noexcept override {
    return "NanoRowSoftmax";
  }
  const char* getPluginVersion() const noexcept override { return "1"; }
  const PluginFieldCollection* getFieldNames() noexcept override { return &f; }
  IPluginV2* createPlugin(const char*,
                          const PluginFieldCollection*) noexcept override {
    return nullptr;
  }
  IPluginV2* deserializePlugin(const char*, const void* d,
                               size_t s) noexcept override {
    auto* p = new RowSoftmax();
    int q[3];
    std::memcpy(q, d, s);
    p->restore(q[0], q[1], q[2]);
    p->setPluginNamespace(ns.c_str());
    return p;
  }
  void setPluginNamespace(const char* n) noexcept override { ns = n ? n : ""; }
  const char* getPluginNamespace() const noexcept override {
    return ns.c_str();
  }
};
static SCreator creator;
extern "C" void nano_register_softmax() {
  static bool r = getPluginRegistry()->registerCreator(creator, "");
  (void)r;
}
extern "C" IPluginV2* nano_make_softmax() { return new RowSoftmax(); }
