// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
// See the repository LICENSE for the full license text.

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <string>
using namespace nvinfer1;

// input [4,1,L,2L-1] -> output [4,1,L,L], out[h,0,q,k] = in[h,0,q,L-1-q+k]
template <class T>
__global__ void relpos4d(const T* in, T* out, int total, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int h = i / (L * L), rem = i % (L * L), q = rem / L, k = rem % L;
  int src = (h * L + q) * (2 * L - 1) + (L - 1 - q + k);
  out[i] = in[src];
}
class RelPos4D : public IPluginV2 {
  int L = 0;
  DataType dtype = DataType::kFLOAT;
  std::string ns;

 public:
  RelPos4D(int l) : L(l) {}
  RelPos4D(const void* d, size_t s) {
    int p[2];
    if (s != sizeof(p)) {
      L = 0;
      return;
    }
    std::memcpy(p, d, sizeof(p));
    L = p[0];
    dtype = (DataType)p[1];
  }
  void restore(int l, int dt) {
    L = l;
    dtype = (DataType)dt;
  }
  const char* getPluginType() const noexcept override { return "NanoRelPos4D"; }
  const char* getPluginVersion() const noexcept override { return "1"; }
  int getNbOutputs() const noexcept override { return 1; }
  Dims getOutputDimensions(int, Dims const*, int) noexcept override {
    Dims d{};
    d.nbDims = 4;
    d.d[0] = 4;
    d.d[1] = 1;
    d.d[2] = L;
    d.d[3] = L;
    return d;
  }
  bool supportsFormat(DataType t, PluginFormat f) const noexcept override {
    return (t == DataType::kFLOAT || t == DataType::kHALF) &&
           f == PluginFormat::kNCHW;
  }
  void configureWithFormat(const Dims*, int, const Dims*, int, DataType t,
                           PluginFormat, int) noexcept override {
    dtype = t;
  }
  int initialize() noexcept override { return L > 0 ? 0 : -1; }
  void terminate() noexcept override {}
  size_t getWorkspaceSize(int) const noexcept override { return 0; }
  int enqueue(int batch, const void* const* in, void** out, void*,
              cudaStream_t st) noexcept override {
    int total = batch * 4 * L * L;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (dtype == DataType::kHALF)
      relpos4d<<<grid, block, 0, st>>>((const uint16_t*)in[0],
                                       (uint16_t*)out[0], total, L);
    else
      relpos4d<<<grid, block, 0, st>>>((const float*)in[0], (float*)out[0],
                                       total, L);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
  }
  size_t getSerializationSize() const noexcept override {
    return 2 * sizeof(int);
  }
  void serialize(void* d) const noexcept override {
    int p[2] = {L, (int)dtype};
    std::memcpy(d, p, sizeof(p));
  }
  void destroy() noexcept override { delete this; }
  IPluginV2* clone() const noexcept override {
    auto* p = new RelPos4D(L);
    p->dtype = dtype;
    p->ns = ns;
    return p;
  }
  void setPluginNamespace(const char* n) noexcept override { ns = n ? n : ""; }
  const char* getPluginNamespace() const noexcept override {
    return ns.c_str();
  }
};
class RC : public IPluginCreator {
  std::string ns;
  PluginFieldCollection f{};

 public:
  const char* getPluginName() const noexcept override { return "NanoRelPos4D"; }
  const char* getPluginVersion() const noexcept override { return "1"; }
  const PluginFieldCollection* getFieldNames() noexcept override { return &f; }
  IPluginV2* createPlugin(const char*,
                          const PluginFieldCollection*) noexcept override {
    return nullptr;
  }
  IPluginV2* deserializePlugin(const char*, const void* d,
                               size_t s) noexcept override {
    auto* p = new RelPos4D(d, s);
    p->setPluginNamespace(ns.c_str());
    return p;
  }
  void setPluginNamespace(const char* n) noexcept override { ns = n ? n : ""; }
  const char* getPluginNamespace() const noexcept override {
    return ns.c_str();
  }
};
static RC creator;
extern "C" void nano_register_relpos4d() {
  static bool r = getPluginRegistry()->registerCreator(creator, "");
  (void)r;
}
extern "C" IPluginV2* nano_make_relpos4d(int L) { return new RelPos4D(L); }
