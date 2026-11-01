// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
// See the repository LICENSE for the full license text.

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace nvinfer1;
extern "C" void nano_register_relpos4d();
extern "C" IPluginV2* nano_make_relpos4d(int length);

static int run_one(int L, int half) {
  int count = 4 * L * L, in_count = 4 * L * (2 * L - 1);
  auto* p = nano_make_relpos4d(L);
  Dims in{};
  in.nbDims = 4;
  in.d[0] = 4;
  in.d[1] = 1;
  in.d[2] = L;
  in.d[3] = 2 * L - 1;
  Dims out = p->getOutputDimensions(0, &in, 1);
  p->configureWithFormat(&in, 1, &out, 1,
                         half ? DataType::kHALF : DataType::kFLOAT,
                         PluginFormat::kNCHW, 1);
  if (p->initialize()) {
    std::printf("INIT_FAIL L=%d half=%d\n", L, half);
    return 1;
  }
  size_t es = half ? 2 : 4;
  std::vector<float> src(in_count);
  for (int i = 0; i < in_count; i++)
    src[i] = float(i % 1000) / 8.0f;  // exactly representable in half
  void *a = nullptr, *b = nullptr;
  if (cudaMalloc(&a, in_count * es) != cudaSuccess ||
      cudaMalloc(&b, count * es) != cudaSuccess)
    return 1;
  if (half) {
    std::vector<__half> h(in_count);
    for (int i = 0; i < in_count; i++) h[i] = __float2half(src[i]);
    cudaMemcpy(a, h.data(), in_count * es, cudaMemcpyHostToDevice);
  } else
    cudaMemcpy(a, src.data(), in_count * es, cudaMemcpyHostToDevice);
  const void* ins[] = {a};
  void* outs[] = {b};
  if (p->enqueue(1, ins, outs, nullptr, nullptr)) {
    std::printf("ENQUEUE_FAIL L=%d half=%d\n", L, half);
    return 1;
  }
  cudaDeviceSynchronize();
  std::vector<float> got(count);
  if (half) {
    std::vector<__half> h(count);
    cudaMemcpy(h.data(), b, count * es, cudaMemcpyDeviceToHost);
    for (int i = 0; i < count; i++) got[i] = __half2float(h[i]);
  } else
    cudaMemcpy(got.data(), b, count * es, cudaMemcpyDeviceToHost);
  int bad = 0, nan = 0;
  for (int i = 0; i < count; i++) {
    if (!std::isfinite(got[i])) {
      nan++;
      continue;
    }
    int h = i / (L * L), rem = i % (L * L), q = rem / L, k = rem % L;
    int correct = (h * L + q) * (2 * L - 1) + (L - 1 - q + k);
    if (got[i] != src[correct]) bad++;
  }
  p->terminate();
  p->destroy();
  cudaFree(a);
  cudaFree(b);
  bool ok = (bad == 0 && nan == 0);
  std::printf(
      "{\"L\":%d,\"half\":%d,\"elements\":%d,\"nan\":%d,\"mismatch\":%d,"
      "\"pass\":%s}\n",
      L, half, count, nan, bad, ok ? "true" : "false");
  return ok ? 0 : 1;
}
int main() {
  nano_register_relpos4d();
  int rc = 0;
  for (int L : {4, 256, 512, 1024})
    for (int half = 0; half < 2; half++) rc |= run_one(L, half);
  return rc ? 1 : 0;
}
