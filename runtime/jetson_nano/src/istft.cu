// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
// See the repository LICENSE for the full license text.

#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <cstdio>
#include <new>
// Periodic Hann, centered 1024-point inverse STFT; no normalization in cuFFT.
// Each output thread sums frames in ascending order, matching the CPU oracle.
struct Istft {
  int frames = 0, count = 0;
  cudaStream_t stream = nullptr;
  cufftHandle plan = 0;
  float *mag = nullptr, *cosine = nullptr, *sine = nullptr, *wave = nullptr,
        *window = nullptr, *out = nullptr;
  cufftComplex* spec = nullptr;
};
static thread_local char last_error[256];
static int fail(const char* s, int n) {
  snprintf(last_error, sizeof(last_error), "%s: %d", s, n);
  return -1;
}
extern "C" const char* nano_istft_error() { return last_error; }
__global__ void pack_spectrum(const float* m, const float* c, const float* s,
                              cufftComplex* z, int F) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= F * 513) return;
  int f = i / 513, k = i % 513, j = k * F + f;
  z[i].x = __fmul_rn(m[j], c[j]);
  z[i].y = (k == 0 || k == 512) ? 0.f : __fmul_rn(m[j], s[j]);
}
__global__ void overlap_hann(const float* wave, const float* w, float* out,
                             int F, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int p = i + 512;
  int lo = max(0, (p - 1024 + 256) / 256), hi = min(F - 1, p / 256);
  float sum = 0, den = 0;
  for (int f = lo; f <= hi; f++) {
    int k = p - f * 256;
    float win = w[k];
    float v = __fmul_rn(wave[f * 1024 + k], 1.f / 1024.f);
    sum = __fadd_rn(sum, __fmul_rn(v, win));
    den = __fadd_rn(den, __fmul_rn(win, win));
  }
  out[i] = den ? sum / den : sum;
}
extern "C" void nano_istft_destroy(void* v) {
  auto* p = (Istft*)v;
  if (!p) return;
  if (p->stream) cudaStreamSynchronize(p->stream);
  if (p->plan) cufftDestroy(p->plan);
  void* buffers[] = {p->mag,    p->cosine, p->sine, p->wave,
                     p->window, p->out,    p->spec};
  for (void* q : buffers)
    if (q) cudaFree(q);
  if (p->stream) cudaStreamDestroy(p->stream);
  delete p;
}
extern "C" void* nano_istft_create(int F) {
  last_error[0] = 0;
  if (F < 2) {
    fail("frames", F);
    return nullptr;
  }
  auto* p = new (std::nothrow) Istft;
  if (!p) return nullptr;
  p->frames = F;
  p->count = (F - 1) * 256;
#define CUDA_INIT(expr)      \
  do {                       \
    auto e = (expr);         \
    if (e != cudaSuccess) {  \
      fail(#expr, (int)e);   \
      nano_istft_destroy(p); \
      return nullptr;        \
    }                        \
  } while (0)
  CUDA_INIT(cudaStreamCreateWithFlags(&p->stream, cudaStreamNonBlocking));
  CUDA_INIT(cudaMalloc(&p->mag, F * 513 * sizeof(float)));
  CUDA_INIT(cudaMalloc(&p->cosine, F * 513 * sizeof(float)));
  CUDA_INIT(cudaMalloc(&p->sine, F * 513 * sizeof(float)));
  CUDA_INIT(cudaMalloc(&p->spec, F * 513 * sizeof(cufftComplex)));
  CUDA_INIT(cudaMalloc(&p->wave, F * 1024 * sizeof(float)));
  CUDA_INIT(cudaMalloc(&p->window, 1024 * sizeof(float)));
  CUDA_INIT(cudaMalloc(&p->out, p->count * sizeof(float)));
  float w[1024];
  for (int k = 0; k < 1024; k++)
    w[k] = (float)(.5 -
                   .5 * std::cos(6.283185307179586476925286766559 * k / 1024));
  CUDA_INIT(cudaMemcpy(p->window, w, sizeof(w), cudaMemcpyHostToDevice));
  auto e = cufftPlan1d(&p->plan, 1024, CUFFT_C2R, F);
  if (e != CUFFT_SUCCESS) {
    fail("cufftPlan1d", e);
    nano_istft_destroy(p);
    return nullptr;
  }
  e = cufftSetStream(p->plan, p->stream);
  if (e != CUFFT_SUCCESS) {
    fail("cufftSetStream", e);
    nano_istft_destroy(p);
    return nullptr;
  }
  return p;
}
extern "C" int nano_istft_run(void* v, const float* m, const float* c,
                              const float* s, float* out) {
  auto* p = (Istft*)v;
  if (!p || !m || !c || !s || !out) return fail("null input", 0);
  size_t bytes = p->frames * 513 * sizeof(float);
#define CUDA_RUN(expr)                                \
  do {                                                \
    auto e = (expr);                                  \
    if (e != cudaSuccess) return fail(#expr, (int)e); \
  } while (0)
  CUDA_RUN(
      cudaMemcpyAsync(p->mag, m, bytes, cudaMemcpyHostToDevice, p->stream));
  CUDA_RUN(
      cudaMemcpyAsync(p->cosine, c, bytes, cudaMemcpyHostToDevice, p->stream));
  CUDA_RUN(
      cudaMemcpyAsync(p->sine, s, bytes, cudaMemcpyHostToDevice, p->stream));
  pack_spectrum<<<(p->frames * 513 + 255) / 256, 256, 0, p->stream>>>(
      p->mag, p->cosine, p->sine, p->spec, p->frames);
  CUDA_RUN(cudaPeekAtLastError());
  auto e = cufftExecC2R(p->plan, p->spec, p->wave);
  if (e != CUFFT_SUCCESS) return fail("cufftExecC2R", e);
  overlap_hann<<<(p->count + 255) / 256, 256, 0, p->stream>>>(
      p->wave, p->window, p->out, p->frames, p->count);
  CUDA_RUN(cudaPeekAtLastError());
  CUDA_RUN(cudaMemcpyAsync(out, p->out, p->count * sizeof(float),
                           cudaMemcpyDeviceToHost, p->stream));
  CUDA_RUN(cudaStreamSynchronize(p->stream));
  return 0;
}
