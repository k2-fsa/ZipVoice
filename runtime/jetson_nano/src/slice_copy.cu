// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
// See the repository LICENSE for the full license text.

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
using namespace nvinfer1;

// out[row, k] = in[row, idx[k]]; input flattened as [rows, C], output [rows,
// K].
template <class T>
__global__ void slice_copy(const T* in, T* out, int total, int K, int C,
                           const int* idx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int row = i / K;
  int col = i - row * K;
  out[i] = in[(long)row * C + idx[col]];
}

namespace {
class SliceCopy : public IPluginV2 {
  int K = 0, rows = 0, C = 0;
  DataType dtype = DataType::kFLOAT;
  std::string ns;
  std::vector<int> host_idx;
  int* dev_idx = nullptr;

 public:
  SliceCopy(int k, int c, const int* idx) : K(k), C(c) {
    host_idx.assign(idx, idx + k);
  }
  SliceCopy(const void* data, size_t size) {
    const char* p = static_cast<const char*>(data);
    const char* end = p + size;
    if (size < 4 * sizeof(int)) {
      K = 0;
      return;
    }
    std::memcpy(&K, p, 4);
    p += 4;
    std::memcpy(&rows, p, 4);
    p += 4;
    std::memcpy(&C, p, 4);
    p += 4;
    int dt;
    std::memcpy(&dt, p, 4);
    p += 4;
    dtype = static_cast<DataType>(dt);
    if (K < 0 || rows <= 0 || C <= 0 || p + 4LL * K > end) {
      K = 0;
      return;
    }
    host_idx.resize(K);
    std::memcpy(host_idx.data(), p, 4 * K);
  }
  const char* getPluginType() const noexcept override {
    return "NanoSliceCopy";
  }
  const char* getPluginVersion() const noexcept override { return "1"; }
  int getNbOutputs() const noexcept override { return 1; }
  Dims getOutputDimensions(int, Dims const* inputs, int) noexcept override {
    Dims d = inputs[0];
    if (d.nbDims >= 1) d.d[d.nbDims - 1] = K;
    return d;
  }
  bool supportsFormat(DataType t, PluginFormat f) const noexcept override {
    return (t == DataType::kFLOAT || t == DataType::kHALF) &&
           f == PluginFormat::kNCHW;
  }
  void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims*,
                           int, DataType t, PluginFormat,
                           int) noexcept override {
    dtype = t;
    if (nbInputs < 1 || inputDims[0].nbDims < 1) return;
    const Dims& d = inputDims[0];
    C = d.d[d.nbDims - 1];
    long r = 1;
    for (int i = 0; i < d.nbDims - 1; i++) r *= d.d[i];
    rows = (int)r;
  }
  int initialize() noexcept override {
    if (K <= 0 || C <= 0 || rows <= 0) return -1;
    if (cudaMalloc(&dev_idx, (size_t)K * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMemcpy(dev_idx, host_idx.data(), (size_t)K * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      return -1;
    return 0;
  }
  void terminate() noexcept override {
    if (dev_idx) {
      cudaFree(dev_idx);
      dev_idx = nullptr;
    }
  }
  size_t getWorkspaceSize(int) const noexcept override { return 0; }
  int enqueue(int batch, const void* const* in, void** out, void*,
              cudaStream_t stream) noexcept override {
    long total = (long)batch * rows * K;
    int block = 256;
    long grid = (total + block - 1) / block;
    if (dtype == DataType::kHALF)
      slice_copy<<<(int)grid, block, 0, stream>>>(
          (const uint16_t*)in[0], (uint16_t*)out[0], (int)total, K, C, dev_idx);
    else
      slice_copy<<<(int)grid, block, 0, stream>>>(
          (const float*)in[0], (float*)out[0], (int)total, K, C, dev_idx);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
  }
  size_t getSerializationSize() const noexcept override {
    return 4 * sizeof(int) + 4ULL * K;
  }
  void serialize(void* data) const noexcept override {
    char* p = static_cast<char*>(data);
    std::memcpy(p, &K, 4);
    p += 4;
    std::memcpy(p, &rows, 4);
    p += 4;
    std::memcpy(p, &C, 4);
    p += 4;
    int dt = static_cast<int>(dtype);
    std::memcpy(p, &dt, 4);
    p += 4;
    std::memcpy(p, host_idx.data(), 4ULL * K);
  }
  void destroy() noexcept override { delete this; }
  IPluginV2* clone() const noexcept override {
    auto* p = new SliceCopy(K, C, host_idx.data());
    p->rows = rows;
    p->dtype = dtype;
    p->ns = ns;
    return p;
  }
  void setPluginNamespace(const char* name) noexcept override {
    ns = name ? name : "";
  }
  const char* getPluginNamespace() const noexcept override {
    return ns.c_str();
  }
};
class Creator : public IPluginCreator {
  std::string ns;
  PluginFieldCollection fields{};

 public:
  const char* getPluginName() const noexcept override {
    return "NanoSliceCopy";
  }
  const char* getPluginVersion() const noexcept override { return "1"; }
  const PluginFieldCollection* getFieldNames() noexcept override {
    return &fields;
  }
  IPluginV2* createPlugin(const char*,
                          const PluginFieldCollection*) noexcept override {
    return nullptr;
  }
  IPluginV2* deserializePlugin(const char*, const void* data,
                               size_t size) noexcept override {
    auto* p = new SliceCopy(data, size);
    p->setPluginNamespace(ns.c_str());
    return p;
  }
  void setPluginNamespace(const char* name) noexcept override {
    ns = name ? name : "";
  }
  const char* getPluginNamespace() const noexcept override {
    return ns.c_str();
  }
};
static Creator creator;
}  // namespace
extern "C" void nano_register_slice() {
  static bool registered = getPluginRegistry()->registerCreator(creator, "");
  (void)registered;
}
extern "C" IPluginV2* nano_make_slice(int K, int C, const int* idx) {
  return new SliceCopy(K, C, idx);
}
extern "C" int nano_check_creator(const char* type) {
  return getPluginRegistry()->getPluginCreator(type, "1") ? 1 : 0;
}
