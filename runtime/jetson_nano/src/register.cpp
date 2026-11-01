// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
#include <NvInfer.h>
#include <NvInferVersion.h>
#if NV_TENSORRT_MAJOR != 6
#error \
    "This optional runtime targets TensorRT 6. Use the regular exporter for newer TensorRT."
#endif
extern "C" void nano_register_relpos();
extern "C" void nano_register_relpos4d();
extern "C" void nano_register_slice();
extern "C" void nano_register_softmax();
extern "C" void nano_register_swoosh();
extern "C" void nano_register_fusions();
extern "C" void nano_register_all() {
  nano_register_relpos();
  nano_register_relpos4d();
  nano_register_slice();
  nano_register_softmax();
  nano_register_swoosh();
  nano_register_fusions();
}
