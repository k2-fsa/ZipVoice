// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
// See the repository LICENSE for the full license text.

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>
using namespace nvinfer1;
template <class T>
__global__ void attn_softmax(const T* score, const T* relative, T* out, int K) {
  int row = blockIdx.x, q = row % K;
  const T* x = score + (size_t)row * K;
  const T* r = relative + (size_t)row * (2 * K - 1);
  T* y = out + (size_t)row * K;
  float v[8], mx = -INFINITY;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int k = threadIdx.x + j * 128;
    v[j] =
        k < K ? (float)(T)((float)x[k] + (float)r[K - 1 - q + k]) : -INFINITY;
    mx = fmaxf(mx, v[j]);
  }
  for (int d = 16; d; d >>= 1)
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
  __shared__ float sh[4];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (!lane) sh[warp] = mx;
  __syncthreads();
  mx = fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3]));
  float s = 0;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    v[j] = expf(v[j] - mx);
    s += v[j];
  }
  for (int d = 16; d; d >>= 1) s += __shfl_xor_sync(0xffffffff, s, d);
  __syncthreads();
  if (!lane) sh[warp] = s;
  __syncthreads();
  s = 1.f / (sh[0] + sh[1] + sh[2] + sh[3]);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int k = threadIdx.x + j * 128;
    if (k < K) y[k] = (T)(v[j] * s);
  }
}
template <class T>
__global__ void fused_glu(const T* x, T* y, int count, int K) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    int r = i / K, c = i % K;
    T gate = (T)(1.f / (1.f + expf(-(float)x[(size_t)r * (2 * K) + K + c])));
    y[i] = (T)((float)x[(size_t)r * (2 * K) + c] * (float)gate);
  }
}

template <class T, int K>
__global__ void attn_softmax_fixed(const T* score, const T* relative, T* out) {
  constexpr int N = K / 128;
  int row = blockIdx.x, q = row & (K - 1);
  const T* x = score + (size_t)row * K;
  const T* r = relative + (size_t)row * (2 * K - 1);
  T* y = out + (size_t)row * K;
  float v[N], mx = -INFINITY;
#pragma unroll
  for (int j = 0; j < N; j++) {
    int k = threadIdx.x + j * 128;
    v[j] = (float)(T)((float)x[k] + (float)r[K - 1 - q + k]);
    mx = fmaxf(mx, v[j]);
  }
  for (int d = 16; d; d >>= 1)
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
  __shared__ float sh[4];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (!lane) sh[warp] = mx;
  __syncthreads();
  mx = fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3]));
  float sum = 0;
#pragma unroll
  for (int j = 0; j < N; j++) {
    v[j] = expf(v[j] - mx);
    sum += v[j];
  }
  for (int d = 16; d; d >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, d);
  __syncthreads();
  if (!lane) sh[warp] = sum;
  __syncthreads();
  sum = 1.f / (sh[0] + sh[1] + sh[2] + sh[3]);
#pragma unroll
  for (int j = 0; j < N; j++) y[threadIdx.x + j * 128] = (T)(v[j] * sum);
}
__device__ float2 load_pair(const float* x) { return *(const float2*)x; }
__device__ float2 load_pair(const __half* x) {
  return __half22float2(*(const __half2*)x);
}
__device__ void store_pair(float* x, float2 v) { *(float2*)x = v; }
__device__ void store_pair(__half* x, float2 v) {
  *(__half2*)x = __floats2half2_rn(v.x, v.y);
}
template <class T>
__global__ void glu_fixed512(const T* x, T* y, int pairs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pairs) return;
  int base = (i >> 8) * 1024 + ((i & 255) << 1);
  float2 a = load_pair(x + base), z = load_pair(x + base + 512);
  T g0 = (T)(1.f / (1.f + expf(-z.x))), g1 = (T)(1.f / (1.f + expf(-z.y)));
  store_pair(y + 2 * i, make_float2(a.x * (float)g0, a.y * (float)g1));
}
template <class T>
void launch_exact_attention(const void* const* in, void** out, int rows, int K,
                            cudaStream_t st) {
  if (K == 256)
    attn_softmax_fixed<T, 256>
        <<<rows, 128, 0, st>>>((const T*)in[0], (const T*)in[1], (T*)out[0]);
  else if (K == 512)
    attn_softmax_fixed<T, 512>
        <<<rows, 128, 0, st>>>((const T*)in[0], (const T*)in[1], (T*)out[0]);
  else if (K == 1024)
    attn_softmax_fixed<T, 1024>
        <<<rows, 128, 0, st>>>((const T*)in[0], (const T*)in[1], (T*)out[0]);
  else
    attn_softmax<<<rows, 128, 0, st>>>((const T*)in[0], (const T*)in[1],
                                       (T*)out[0], K);
}

class NanoSpecializedFused : public IPluginV2 {
 public:
  int mode, rows = 0, K = 0;
  DataType dt = DataType::kFLOAT;
  std::string ns;
  NanoSpecializedFused(int m) : mode(m) {}
  NanoSpecializedFused(const void* d, size_t n) {
    int v[4] = {0, 0, 0, 0};
    if (n == sizeof(v)) memcpy(v, d, sizeof(v));
    mode = v[0];
    rows = v[1];
    K = v[2];
    dt = (DataType)v[3];
  }
  const char* getPluginType() const noexcept override {
    return mode ? "NanoFusedGLU" : "NanoAttentionSoftmax";
  }
  const char* getPluginVersion() const noexcept override { return "1"; }
  int getNbOutputs() const noexcept override { return 1; }
  Dims getOutputDimensions(int, const Dims* in, int) noexcept override {
    Dims d = in[0];
    if (mode) d.d[d.nbDims - 1] /= 2;
    return d;
  }
  bool supportsFormat(DataType t, PluginFormat f) const noexcept override {
    return (t == DataType::kFLOAT || t == DataType::kHALF) &&
           f == PluginFormat::kNCHW;
  }
  void configureWithFormat(const Dims* in, int, const Dims*, int, DataType t,
                           PluginFormat, int) noexcept override {
    dt = t;
    K = in[0].d[in[0].nbDims - 1] / (mode ? 2 : 1);
    rows = 1;
    for (int j = 0; j < in[0].nbDims - 1; j++) rows *= in[0].d[j];
  }
  int initialize() noexcept override {
    return rows > 0 && K > 0 && (mode || K <= 1024) ? 0 : -1;
  }
  void terminate() noexcept override {}
  size_t getWorkspaceSize(int) const noexcept override { return 0; }
  int enqueue(int b, const void* const* in, void** out, void*,
              cudaStream_t st) noexcept override {
    if (mode) {
      int n = b * rows * K;
      if (K == 512 && n % 2 == 0) {
        if (dt == DataType::kHALF)
          glu_fixed512<<<(n / 2 + 255) / 256, 256, 0, st>>>(
              (const __half*)in[0], (__half*)out[0], n / 2);
        else
          glu_fixed512<<<(n / 2 + 255) / 256, 256, 0, st>>>(
              (const float*)in[0], (float*)out[0], n / 2);
      } else if (dt == DataType::kHALF)
        fused_glu<<<(n + 255) / 256, 256, 0, st>>>((const __half*)in[0],
                                                   (__half*)out[0], n, K);
      else
        fused_glu<<<(n + 255) / 256, 256, 0, st>>>((const float*)in[0],
                                                   (float*)out[0], n, K);
    } else {
      if (dt == DataType::kHALF)
        launch_exact_attention<__half>(in, out, b * rows, K, st);
      else
        launch_exact_attention<float>(in, out, b * rows, K, st);
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
  }
  size_t getSerializationSize() const noexcept override {
    return 4 * sizeof(int);
  }
  void serialize(void* d) const noexcept override {
    int v[4] = {mode, rows, K, (int)dt};
    memcpy(d, v, sizeof(v));
  }
  void destroy() noexcept override { delete this; }
  IPluginV2* clone() const noexcept override {
    auto* p = new NanoSpecializedFused(mode);
    p->rows = rows;
    p->K = K;
    p->dt = dt;
    p->ns = ns;
    return p;
  }
  void setPluginNamespace(const char* n) noexcept override { ns = n ? n : ""; }
  const char* getPluginNamespace() const noexcept override {
    return ns.c_str();
  }
};
class SpecializedFusedCreator : public IPluginCreator {
  int mode;
  std::string ns;
  PluginFieldCollection f{};

 public:
  SpecializedFusedCreator(int m) : mode(m) {}
  const char* getPluginName() const noexcept override {
    return mode ? "NanoFusedGLU" : "NanoAttentionSoftmax";
  }
  const char* getPluginVersion() const noexcept override { return "1"; }
  const PluginFieldCollection* getFieldNames() noexcept override { return &f; }
  IPluginV2* createPlugin(const char*,
                          const PluginFieldCollection*) noexcept override {
    return nullptr;
  }
  IPluginV2* deserializePlugin(const char*, const void* d,
                               size_t n) noexcept override {
    auto* p = new NanoSpecializedFused(d, n);
    p->setPluginNamespace(ns.c_str());
    return p;
  }
  void setPluginNamespace(const char* n) noexcept override { ns = n ? n : ""; }
  const char* getPluginNamespace() const noexcept override {
    return ns.c_str();
  }
};
static SpecializedFusedCreator attn_creator(0), glu_creator(1);
extern "C" void nano_register_fusions() {
  static bool a = getPluginRegistry()->registerCreator(attn_creator, ""),
              b = getPluginRegistry()->registerCreator(glu_creator, "");
  (void)a;
  (void)b;
}
extern "C" IPluginV2* nano_make_attention_softmax() {
  return new NanoSpecializedFused(0);
}
extern "C" IPluginV2* nano_make_glu() { return new NanoSpecializedFused(1); }
