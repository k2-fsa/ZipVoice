// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
// See the repository LICENSE for the full license text.

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
using namespace nvinfer1;
extern "C" void nano_register_fusions();
extern "C" IPluginV2* nano_make_attention_softmax();
extern "C" IPluginV2* nano_make_glu();
extern "C" void nano_register_relpos4d();
extern "C" IPluginV2* nano_make_relpos4d(int);
extern "C" void nano_register_softmax();
extern "C" IPluginV2* nano_make_softmax();
extern "C" void nano_register_slice();
extern "C" IPluginV2* nano_make_slice(int, int, const int*);
class Log : public ILogger {
  void log(Severity s, const char* t) noexcept override {
    if (s <= Severity::kWARNING) fprintf(stderr, "%s\n", t);
  }
};
static std::vector<IRuntime*> runtimes;
ICudaEngine* build(Log& log, int mode, int L, DataType dt, bool fused) {
  auto* b = createInferBuilder(log);
  auto* n = b->createNetworkV2(0);
  Dims d{}, rd{};
  std::vector<IPluginV2*> plugins;
  ITensor* out;
  if (mode == 0) {
    d.nbDims = rd.nbDims = 4;
    d.d[0] = rd.d[0] = 4;
    d.d[1] = rd.d[1] = 1;
    d.d[2] = rd.d[2] = L;
    d.d[3] = L;
    rd.d[3] = 2 * L - 1;
    auto* x = n->addInput("x", dt, d);
    auto* r = n->addInput("r", dt, rd);
    if (fused) {
      auto* p = nano_make_attention_softmax();
      plugins.push_back(p);
      ITensor* ins[] = {x, r};
      out = n->addPluginV2(ins, 2, *p)->getOutput(0);
    } else {
      auto* p = nano_make_relpos4d(L);
      plugins.push_back(p);
      ITensor* ins[] = {r};
      auto* copy = n->addPluginV2(ins, 1, *p);
      copy->setPrecision(dt);
      copy->setOutputType(0, dt);
      auto* sum = n->addElementWise(*x, *copy->getOutput(0),
                                    ElementWiseOperation::kSUM);
      sum->setPrecision(dt);
      sum->setOutputType(0, dt);
      auto* s = nano_make_softmax();
      plugins.push_back(s);
      ITensor* sm[] = {sum->getOutput(0)};
      out = n->addPluginV2(sm, 1, *s)->getOutput(0);
    }
  } else {
    d.nbDims = 2;
    d.d[0] = L;
    d.d[1] = 1024;
    auto* x = n->addInput("x", dt, d);
    if (fused) {
      auto* p = nano_make_glu();
      plugins.push_back(p);
      ITensor* ins[] = {x};
      out = n->addPluginV2(ins, 1, *p)->getOutput(0);
    } else {
      std::vector<int> a(512), z(512);
      for (int i = 0; i < 512; i++) {
        a[i] = i;
        z[i] = i + 512;
      }
      auto* p = nano_make_slice(512, 1024, a.data());
      auto* q = nano_make_slice(512, 1024, z.data());
      plugins.push_back(p);
      plugins.push_back(q);
      ITensor* ins[] = {x};
      auto* l = n->addPluginV2(ins, 1, *p);
      auto* r = n->addPluginV2(ins, 1, *q);
      auto* s = n->addActivation(*r->getOutput(0), ActivationType::kSIGMOID);
      s->setPrecision(dt);
      s->setOutputType(0, dt);
      auto* m = n->addElementWise(*l->getOutput(0), *s->getOutput(0),
                                  ElementWiseOperation::kPROD);
      m->setPrecision(dt);
      m->setOutputType(0, dt);
      out = m->getOutput(0);
    }
  }
  out->setName("y");
  out->setType(dt);
  n->markOutput(*out);
  b->setMaxBatchSize(1);
  b->setMaxWorkspaceSize(4 << 20);
  b->setFp16Mode(dt == DataType::kHALF);
  b->setMinFindIterations(1);
  b->setAverageFindIterations(1);
  auto* e = b->buildCudaEngine(*n);
  if (!e) throw std::runtime_error("build");
  if (fused) {
    auto* blob = e->serialize();
    auto* r = createInferRuntime(log);
    auto* e2 = r->deserializeCudaEngine(blob->data(), blob->size(), nullptr);
    if (!e2) throw std::runtime_error("deserialize");
    e->destroy();
    e = e2;
    blob->destroy();
    runtimes.push_back(r);
  }
  n->destroy();
  b->destroy();
  for (auto* p : plugins) p->destroy();
  return e;
}
struct Run {
  ICudaEngine* e;
  IExecutionContext* c;
  std::vector<void*> p;
  Run(ICudaEngine* x) : e(x) {
    c = e->createExecutionContext();
    p.resize(e->getNbBindings());
    for (int j = 0; j < e->getNbBindings(); j++) {
      auto d = e->getBindingDimensions(j);
      size_t count = 1;
      for (int k = 0; k < d.nbDims; k++) count *= d.d[k];
      cudaMalloc(&p[j],
                 count * (e->getBindingDataType(j) == DataType::kHALF ? 2 : 4));
    }
  }
  ~Run() {
    for (auto* x : p) cudaFree(x);
    c->destroy();
    e->destroy();
  }
  float time() {
    cudaEvent_t a, z;
    cudaEventCreate(&a);
    cudaEventCreate(&z);
    cudaEventRecord(a);
    for (int j = 0; j < 8; j++)
      if (!c->enqueue(1, p.data(), 0, nullptr))
        throw std::runtime_error("enqueue");
    cudaEventRecord(z);
    cudaEventSynchronize(z);
    float ms;
    cudaEventElapsedTime(&ms, a, z);
    cudaEventDestroy(a);
    cudaEventDestroy(z);
    return ms / 8;
  }
};
template <class T>
bool test(Log& log, int mode, int L, DataType dt) {
  Run a(build(log, mode, L, dt, false)), b(build(log, mode, L, dt, true));
  size_t nx = mode ? (size_t)L * 1024 : (size_t)4 * L * L,
         nr = mode ? 0 : (size_t)4 * L * (2 * L - 1),
         ny = mode ? (size_t)L * 512 : nx;
  std::vector<T> x(nx), r(nr), base(ny), cand(ny);
  std::mt19937 gen(73);
  std::normal_distribution<float> norm(0, 2);
  for (auto& v : x) v = (T)norm(gen);
  for (auto& v : r) v = (T)norm(gen);
  for (Run* p : {&a, &b}) {
    cudaMemcpy(p->p[p->e->getBindingIndex("x")], x.data(), nx * sizeof(T),
               cudaMemcpyHostToDevice);
    if (!mode)
      cudaMemcpy(p->p[p->e->getBindingIndex("r")], r.data(), nr * sizeof(T),
                 cudaMemcpyHostToDevice);
    p->time();
  }
  std::vector<float> ta, tb;
  for (int j = 0; j < 3; j++) {
    ta.push_back(a.time());
    tb.push_back(b.time());
  }
  std::sort(ta.begin(), ta.end());
  std::sort(tb.begin(), tb.end());
  cudaMemcpy(base.data(), a.p[a.e->getBindingIndex("y")], ny * sizeof(T),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cand.data(), b.p[b.e->getBindingIndex("y")], ny * sizeof(T),
             cudaMemcpyDeviceToHost);
  double err = 0, den = 0, gerr = 0, gden = 0;
  bool finite = true;
  if (mode) {
    for (size_t i = 0; i < ny; i++) {
      size_t row = i / 512, k = i % 512;
      T gate = (T)(1.f / (1.f + expf(-(float)x[row * 1024 + 512 + k])));
      double gold = (float)(T)((float)x[row * 1024 + k] * (float)gate),
             v = (float)cand[i], ref = (float)base[i];
      finite &= std::isfinite(v);
      err += (v - ref) * (v - ref);
      den += ref * ref;
      gerr += (v - gold) * (v - gold);
      gden += gold * gold;
    }
  } else {
    for (int row = 0; row < 4 * L; row++) {
      double mx = -INFINITY;
      for (int k = 0; k < L; k++) {
        T t = (T)((float)x[(size_t)row * L + k] +
                  (float)r[(size_t)row * (2 * L - 1) + L - 1 - row % L + k]);
        mx = fmax(mx, (float)t);
      }
      double sum = 0;
      for (int k = 0; k < L; k++) {
        T t = (T)((float)x[(size_t)row * L + k] +
                  (float)r[(size_t)row * (2 * L - 1) + L - 1 - row % L + k]);
        sum += exp((float)t - mx);
      }
      for (int k = 0; k < L; k++) {
        size_t i = (size_t)row * L + k;
        T t = (T)((float)x[i] +
                  (float)r[(size_t)row * (2 * L - 1) + L - 1 - row % L + k]);
        double gold = exp((float)t - mx) / sum, v = (float)cand[i],
               ref = (float)base[i];
        finite &= std::isfinite(v);
        err += (v - ref) * (v - ref);
        den += ref * ref;
        gerr += (v - gold) * (v - gold);
        gden += gold * gold;
      }
    }
  }
  double rel = sqrt(err / den), goldrel = sqrt(gerr / gden);
  printf(
      "{\"mode\":%d,\"L\":%d,\"dtype\":%d,\"baseline_ms\":%.6f,\"candidate_"
      "ms\":%.6f,\"speedup\":%.6f,\"relative_rmse\":%.9f,\"cpu_relative_rmse\":"
      "%.9f,\"finite\":%s}\n",
      mode, L, (int)dt, ta[1], tb[1], ta[1] / tb[1], rel, goldrel,
      finite ? "true" : "false");
  fflush(stdout);
  return finite && rel < .001 && goldrel < .001;
}
int main() {
  try {
    nano_register_fusions();
    nano_register_relpos4d();
    nano_register_softmax();
    nano_register_slice();
    Log log;
    bool ok = true;
    for (int mode : {0, 1})
      for (int L : {256, 512, 1024}) {
        ok = test<__half>(log, mode, L, DataType::kHALF) && ok;
        ok = test<float>(log, mode, L, DataType::kFLOAT) && ok;
      }
    for (auto* r : runtimes) r->destroy();
    return ok ? 0 : 4;
  } catch (const std::exception& e) {
    fprintf(stderr, "ERROR %s\n", e.what());
    return 1;
  }
}
