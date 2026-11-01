// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
// See the repository LICENSE for the full license text.

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
using namespace nvinfer1;

template <class T>
__global__ void swoosh(const T* in, T* out, int n, float c, float a, float b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = (float)in[i], z = x - c;
  float sp = z > 20.f ? z : log1pf(expf(z));
  out[i] = (T)(sp - a * x - b);
}

__global__ void make_swoosh_lut(unsigned short* table, float c, float a,
                                float b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 65536) return;
  float x = __half2float(__ushort_as_half((unsigned short)i)), z = x - c;
  float sp = z > 20.f ? z : log1pf(expf(z));
  table[i] = __half_as_ushort(__float2half_rn(sp - a * x - b));
}
__global__ void swoosh_lut_pairs(const unsigned short* in, unsigned short* out,
                                 const unsigned short* table, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = i * 2;
  if (j + 1 < n) {
    unsigned int v = ((const unsigned int*)in)[i];
    unsigned int lo = __ldg(table + (v & 65535)), hi = __ldg(table + (v >> 16));
    ((unsigned int*)out)[i] = lo | (hi << 16);
  } else if (j < n)
    out[j] = __ldg(table + in[j]);
}

__global__ void swoosh_lut_shared_pairs(const unsigned short* in,
                                        unsigned short* out,
                                        const unsigned short* table, int n) {
  __shared__ unsigned short hot[14336];
  for (int i = threadIdx.x; i < 14336; i += 1024)
    hot[i] = __ldg(table + (i < 7168 ? 0x3000 + i : 0xb000 + i - 7168));
  __syncthreads();
  for (int step = 0; step < 16; step++) {
    int pair = blockIdx.x * 16384 + threadIdx.x + step * 1024, j = pair * 2;
    if (j >= n) continue;
    unsigned int bits = j + 1 < n ? ((const unsigned int*)in)[pair] : in[j];
    unsigned int codes[2] = {bits & 65535, bits >> 16}, values[2];
#pragma unroll
    for (int k = 0; k < 2; k++) {
      unsigned int c = codes[k], a = c & 0x7fff;
      values[k] = (a >= 0x3000 && a < 0x4c00)
                      ? hot[(c & 0x8000 ? 7168 : 0) + a - 0x3000]
                      : __ldg(table + c);
    }
    if (j + 1 < n)
      ((unsigned int*)out)[pair] = values[0] | (values[1] << 16);
    else
      out[j] = values[0];
  }
}

class NanoSharedLutSwooshImpl : public IPluginV2 {
  unsigned short* lut = nullptr;
  float c = 0, a = 0, b = 0;
  int n = 0;
  DataType dtype = DataType::kFLOAT;
  std::string ns;

 public:
  NanoSharedLutSwooshImpl(float cc, float aa, float bb) : c(cc), a(aa), b(bb) {}
  NanoSharedLutSwooshImpl(const void* d, size_t s) {
    float p[5] = {0, 0, 0, 0, 0};
    if (s < sizeof(p)) {
      if (s < 12) return;
      std::memcpy(p, d, s < 20 ? s : 20);
    } else
      std::memcpy(p, d, sizeof(p));
    c = p[0];
    a = p[1];
    b = p[2];
    n = (int)p[3];
    dtype = (DataType)(int)p[4];
  }
  const char* getPluginType() const noexcept override { return "NanoSwoosh"; }
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
    if (nb < 1) return;
    long s = 1;
    for (int i = 0; i < in[0].nbDims; i++) s *= in[0].d[i];
    n = (int)s;
  }
  int initialize() noexcept override {
    if (n <= 0 || n > (1 << 24)) return -1;
    if (dtype != DataType::kHALF) return 0;
    if (cudaMalloc(&lut, 65536 * sizeof(unsigned short)) != cudaSuccess)
      return -1;
    make_swoosh_lut<<<256, 256>>>(lut, c, a, b);
    if (cudaPeekAtLastError() != cudaSuccess) return -1;
    return cudaStreamSynchronize(0) == cudaSuccess ? 0 : -1;
  }
  void terminate() noexcept override {
    if (lut) cudaFree(lut);
    lut = nullptr;
  }
  size_t getWorkspaceSize(int) const noexcept override { return 0; }
  int enqueue(int batch, const void* const* in, void** out, void*,
              cudaStream_t st) noexcept override {
    int total = batch * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (dtype == DataType::kHALF) {
      if (!lut) return -1;
      if (total >= 262144)
        swoosh_lut_shared_pairs<<<(total + 32767) / 32768, 1024, 0, st>>>(
            (const unsigned short*)in[0], (unsigned short*)out[0], lut, total);
      else
        swoosh_lut_pairs<<<((total + 1) / 2 + block - 1) / block, block, 0,
                           st>>>((const unsigned short*)in[0],
                                 (unsigned short*)out[0], lut, total);
    } else
      swoosh<<<grid, block, 0, st>>>((const float*)in[0], (float*)out[0], total,
                                     c, a, b);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
  }
  size_t getSerializationSize() const noexcept override {
    return 5 * sizeof(float);
  }
  void serialize(void* d) const noexcept override {
    float p[5] = {c, a, b, (float)n, (float)(int)dtype};
    std::memcpy(d, p, sizeof(p));
  }
  void destroy() noexcept override { delete this; }
  IPluginV2* clone() const noexcept override {
    auto* p = new NanoSharedLutSwooshImpl(c, a, b);
    p->dtype = dtype;
    p->n = n;
    p->ns = ns;
    return p;
  }
  void setPluginNamespace(const char* n) noexcept override { ns = n ? n : ""; }
  const char* getPluginNamespace() const noexcept override {
    return ns.c_str();
  }
};
class NanoSharedLutSwooshCreator : public IPluginCreator {
  std::string ns;
  PluginFieldCollection f{};

 public:
  const char* getPluginName() const noexcept override { return "NanoSwoosh"; }
  const char* getPluginVersion() const noexcept override { return "1"; }
  const PluginFieldCollection* getFieldNames() noexcept override { return &f; }
  IPluginV2* createPlugin(const char*,
                          const PluginFieldCollection*) noexcept override {
    return nullptr;
  }
  IPluginV2* deserializePlugin(const char*, const void* d,
                               size_t s) noexcept override {
    auto* p = new NanoSharedLutSwooshImpl(d, s);
    p->setPluginNamespace(ns.c_str());
    return p;
  }
  void setPluginNamespace(const char* n) noexcept override { ns = n ? n : ""; }
  const char* getPluginNamespace() const noexcept override {
    return ns.c_str();
  }
};
static NanoSharedLutSwooshCreator creator;
extern "C" void nano_register_swoosh() {
  static bool r = getPluginRegistry()->registerCreator(creator, "");
  (void)r;
}
extern "C" IPluginV2* nano_make_swoosh(float c, float a, float b) {
  return new NanoSharedLutSwooshImpl(c, a, b);
}
