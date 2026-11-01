// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
// See the repository LICENSE for the full license text.

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <string>
using namespace nvinfer1;

template <class T>
__global__ void relative_copy(const T* in, T* out, int length, int shift,
                              int count, int in_count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int row = i >> shift, col = i & (length - 1);
  int source = row * (2 * length - 1) + length - 1 - (row & (length - 1)) + col;
  out[blockIdx.y * count + i] = in[blockIdx.y * in_count + source];
}
extern "C" int nano_relpos_launch(const void* in, void* out, int length,
                                  int half, int batch, cudaStream_t stream) {
  int shift = 0;
  while ((1 << shift) < length) ++shift;
  if ((1 << shift) != length) return -1;
  int count = 4 * length * length, in_count = 4 * length * (2 * length - 1);
  dim3 blocks((count + 255) / 256, batch);
  if (half)
    relative_copy<<<blocks, 256, 0, stream>>>(
        (const uint16_t*)in, (uint16_t*)out, length, shift, count, in_count);
  else
    relative_copy<<<blocks, 256, 0, stream>>>((const float*)in, (float*)out,
                                              length, shift, count, in_count);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}
class RelativeCopy : public IPluginV2 {
  int length;
  DataType dtype = DataType::kFLOAT;
  std::string ns;

 public:
  explicit RelativeCopy(int l) : length(l) {}
  RelativeCopy(const void* data, size_t size) {
    int p[2];
    if (size != sizeof(p)) {
      length = 0;
      return;
    }
    std::memcpy(p, data, sizeof(p));
    length = p[0];
    dtype = static_cast<DataType>(p[1]);
  }
  const char* getPluginType() const noexcept override {
    return "NanoRelativeCopy";
  }
  const char* getPluginVersion() const noexcept override { return "1"; }
  int getNbOutputs() const noexcept override { return 1; }
  Dims getOutputDimensions(int, const Dims*, int) noexcept override {
    Dims d{};
    d.nbDims = 1;
    d.d[0] = 4 * length * length;
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
  int initialize() noexcept override { return length > 0 ? 0 : -1; }
  void terminate() noexcept override {}
  size_t getWorkspaceSize(int) const noexcept override { return 0; }
  int enqueue(int batch, const void* const* in, void** out, void*,
              cudaStream_t stream) noexcept override {
    return nano_relpos_launch(in[0], out[0], length, dtype == DataType::kHALF,
                              batch, stream);
  }
  size_t getSerializationSize() const noexcept override {
    return 2 * sizeof(int);
  }
  void serialize(void* data) const noexcept override {
    int p[] = {length, static_cast<int>(dtype)};
    std::memcpy(data, p, sizeof(p));
  }
  void destroy() noexcept override { delete this; }
  IPluginV2* clone() const noexcept override {
    auto* p = new RelativeCopy(length);
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
    return "NanoRelativeCopy";
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
    auto* p = new RelativeCopy(data, size);
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
extern "C" void nano_register_relpos() {
  static bool registered = getPluginRegistry()->registerCreator(creator, "");
  (void)registered;
}
extern "C" IPluginV2* nano_make_relpos(int length) {
  return new RelativeCopy(length);
}
