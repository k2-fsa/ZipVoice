// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
// See the repository LICENSE for the full license text.

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace nvinfer1;
extern "C" void nano_register_slice();
extern "C" IPluginV2* nano_make_slice(int, int, const int*);
class Log : public ILogger {
  void log(Severity, const char*) noexcept override {}
};
int main() {
  const int B = 1, T = 4, C = 8, K = 3;
  int idx[3] = {0, 3, 7};
  nano_register_slice();
  Log log;
  auto* builder = createInferBuilder(log);
  auto* network = builder->createNetworkV2(0);
  Dims d{};
  d.nbDims = 3;
  d.d[0] = B;
  d.d[1] = T;
  d.d[2] = C;
  auto* in = network->addInput("in", DataType::kFLOAT, d);
  ITensor* ins[] = {in};
  auto* layer = network->addPluginV2(ins, 1, *nano_make_slice(K, C, idx));
  layer->getOutput(0)->setName("out");
  network->markOutput(*layer->getOutput(0));
  builder->setMaxBatchSize(1);
  builder->setMaxWorkspaceSize(1ULL << 20);
  builder->setFp16Mode(true);
  builder->setMinFindIterations(1);
  builder->setAverageFindIterations(1);
  auto* engine = builder->buildCudaEngine(*network);
  if (!engine) {
    std::printf("BUILD_FAIL\n");
    return 1;
  }
  auto* ctx = engine->createExecutionContext();
  std::vector<float> host(B * T * C);
  for (int i = 0; i < B * T * C; i++) host[i] = std::sin(i * 0.37f);
  void *din = nullptr, *dout = nullptr;
  cudaMalloc(&din, B * T * C * 4);
  cudaMalloc(&dout, B * T * K * 4);
  cudaMemcpy(din, host.data(), B * T * C * 4, cudaMemcpyHostToDevice);
  void* bufs[2] = {din, dout};
  if (!ctx->enqueue(1, bufs, nullptr, nullptr)) {
    std::printf("ENQUEUE_FAIL\n");
    return 2;
  }
  cudaDeviceSynchronize();
  std::vector<float> out(B * T * K);
  cudaMemcpy(out.data(), dout, B * T * K * 4, cudaMemcpyDeviceToHost);
  int bad = 0;
  double maxerr = 0;
  for (int b = 0; b < B; b++)
    for (int t = 0; t < T; t++)
      for (int k = 0; k < K; k++) {
        float exp = host[(b * T + t) * C + idx[k]];
        float got = out[(b * T + t) * K + k];
        double e = std::fabs(double(got) - double(exp));
        if (e > maxerr) maxerr = e;
        if (e > 1e-3) bad++;
      }
  std::printf("RESULT bad=%d maxerr=%g first=%g expected=%g\n", bad, maxerr,
              out[0], host[idx[0]]);
  return bad ? 3 : 0;
}
