// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
// See the repository LICENSE for the full license text.

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace nvinfer1;
extern "C" void nano_register_swoosh();
extern "C" IPluginV2* nano_make_swoosh(float, float, float);

void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

// Independent direct evaluation: no LUT or shared-memory implementation here.
template <class T>
__global__ void reference(const T* input, T* output, int n, float c, float a,
                          float b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = (float)input[i], z = x - c;
  float sp = z > 20.f ? z : log1pf(expf(z));
  output[i] = (T)(sp - a * x - b);
}

template <class T>
void run(int n, DataType dtype, float c) {
  const float a = .08001708984375f;
  const float b = c == 1.f ? .313232421875f : .035003662109375f;
  auto* original = nano_make_swoosh(c, a, b);
  Dims dims{};
  dims.nbDims = 1;
  dims.d[0] = n;
  original->configureWithFormat(&dims, 1, &dims, 1, dtype, PluginFormat::kNCHW,
                                1);
  std::vector<char> bytes(original->getSerializationSize());
  original->serialize(bytes.data());
  original->destroy();
  auto* creator = getPluginRegistry()->getPluginCreator("NanoSwoosh", "1", "");
  if (!creator) throw std::runtime_error("Swoosh creator missing");
  auto* plugin = creator->deserializePlugin("test", bytes.data(), bytes.size());
  // TRT 6 may restore without calling configureWithFormat again.
  if (!plugin || plugin->initialize() != 0)
    throw std::runtime_error("Swoosh initialize failed");

  std::vector<T> input(n), actual(n), expected(n);
  std::mt19937 rng(775);
  std::normal_distribution<float> distribution(0, 4);
  for (int i = 0; i < n; ++i) {
    if (sizeof(T) == 2 && n == 65536) {
      unsigned short bits = (unsigned short)i;
      std::memcpy(&input[i], &bits, sizeof(bits));
    } else {
      input[i] = (T)distribution(rng);
    }
  }
  T *x = nullptr, *y = nullptr, *r = nullptr;
  const size_t size = n * sizeof(T);
  check(cudaMalloc(&x, size));
  check(cudaMalloc(&y, size));
  check(cudaMalloc(&r, size));
  check(cudaMemcpy(x, input.data(), size, cudaMemcpyHostToDevice));
  reference<<<(n + 255) / 256, 256>>>(x, r, n, c, a, b);
  check(cudaGetLastError());
  const void* inputs[] = {x};
  void* outputs[] = {y};
  if (plugin->enqueue(1, inputs, outputs, nullptr, 0) != 0)
    throw std::runtime_error("Swoosh enqueue failed");
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  check(cudaMemcpy(actual.data(), y, size, cudaMemcpyDeviceToHost));
  check(cudaMemcpy(expected.data(), r, size, cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (int i = 0; i < n; ++i) {
    // Exhaustive half inputs include NaNs: payload differences are immaterial.
    if (std::isnan((float)actual[i]) && std::isnan((float)expected[i]))
      continue;
    mismatches += std::memcmp(&actual[i], &expected[i], sizeof(T)) != 0;
  }
  plugin->terminate();
  plugin->destroy();
  check(cudaFree(x));
  check(cudaFree(y));
  check(cudaFree(r));
  printf("Swoosh dtype=%s c=%.0f n=%d bit_mismatches=%zu\n",
         dtype == DataType::kHALF ? "half" : "float", c, n, mismatches);
  if (mismatches)
    throw std::runtime_error("Swoosh differs from direct evaluation");
}

int main() {
  try {
    check(cudaFree(nullptr));  // Missing/unusable CUDA must fail the test.
    nano_register_swoosh();
    for (float c : {1.f, 4.f}) {
      for (int n : {65536, 294912, 524288, 1179648, 1966080})
        run<__half>(n, DataType::kHALF, c);
      for (int n : {65536, 524288}) run<float>(n, DataType::kFLOAT, c);
    }
    return 0;
  } catch (const std::exception& error) {
    fprintf(stderr, "Swoosh test failed: %s\n", error.what());
    return 1;
  }
}
