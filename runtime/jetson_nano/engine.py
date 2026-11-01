# Copyright 2026 ZipVoice contributors
# SPDX-License-Identifier: Apache-2.0
"""Small ctypes bridge for the optional TensorRT 6 runtime."""

import ctypes as C
from pathlib import Path

import numpy as np

DEFAULT_LIBRARY = Path(__file__).resolve().parent / "build/libzipvoice_nano.so"


class Engine:
    def __init__(self, filename, library=DEFAULT_LIBRARY):
        library = Path(library).resolve()
        if not library.is_file():
            raise FileNotFoundError(
                "Build the optional Nano runtime first: " + str(library)
            )
        self.lib = C.CDLL(str(library), mode=C.RTLD_GLOBAL)
        self.lib.nano_register_all()
        api = self.lib
        api.nano_trt_error.restype = C.c_char_p
        api.nano_trt_create.argtypes = [C.c_char_p]
        api.nano_trt_create.restype = C.c_void_p
        api.nano_trt_destroy.argtypes = [C.c_void_p]
        api.nano_trt_count.argtypes = [C.c_void_p]
        for name in ["nano_trt_is_input", "nano_trt_dtype", "nano_trt_rank"]:
            getattr(api, name).argtypes = [C.c_void_p, C.c_int]
        api.nano_trt_name.argtypes = [C.c_void_p, C.c_int]
        api.nano_trt_name.restype = C.c_char_p
        api.nano_trt_elements.argtypes = [C.c_void_p, C.c_int]
        api.nano_trt_elements.restype = C.c_size_t
        api.nano_trt_dim.argtypes = [C.c_void_p, C.c_int, C.c_int]
        api.nano_trt_run.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]
        api.nano_trt_run4.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)] + [
            C.c_void_p
        ] * 3
        api.nano_trt_last_ms.argtypes = [C.c_void_p]
        api.nano_trt_last_ms.restype = C.c_float
        self.handle = api.nano_trt_create(str(Path(filename).resolve()).encode())
        if not self.handle:
            raise RuntimeError(api.nano_trt_error().decode())
        self.bindings = []
        self.outputs = {}
        types = {0: np.float32, 1: np.float16, 2: np.int8, 3: np.int32}
        try:
            for i in range(api.nano_trt_count(self.handle)):
                shape = tuple(
                    api.nano_trt_dim(self.handle, i, j)
                    for j in range(api.nano_trt_rank(self.handle, i))
                )
                binding = dict(
                    name=api.nano_trt_name(self.handle, i).decode(),
                    input=bool(api.nano_trt_is_input(self.handle, i)),
                    shape=shape,
                    dtype=types[api.nano_trt_dtype(self.handle, i)],
                )
                self.bindings.append(binding)
                if not binding["input"]:
                    self.outputs[binding["name"]] = np.empty(shape, binding["dtype"])
        except Exception:
            self.close()
            raise

    def _arrays(self, feeds):
        if not self.handle:
            raise RuntimeError("Engine has been closed")
        arrays = []
        for binding in self.bindings:
            if binding["input"]:
                array = np.ascontiguousarray(
                    feeds[binding["name"]], dtype=binding["dtype"]
                )
                if array.shape != binding["shape"]:
                    raise ValueError(
                        "Wrong shape for %s: %s != %s"
                        % (binding["name"], array.shape, binding["shape"])
                    )
            else:
                array = self.outputs[binding["name"]]
            arrays.append(array)
        return arrays

    def run(self, feeds):
        arrays = self._arrays(feeds)
        pointers = (C.c_void_p * len(arrays))(*(a.ctypes.data for a in arrays))
        if self.lib.nano_trt_run(self.handle, pointers):
            raise RuntimeError(self.lib.nano_trt_error().decode())
        return self.outputs

    def run4(self, feeds, steps):
        timestamps = np.asarray(steps, dtype=np.float64)
        if (
            timestamps.shape != (5,)
            or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0)
        ):
            raise ValueError("run4 requires five finite, increasing timestamps")
        times = np.asarray(timestamps, dtype=np.float32)
        delta = np.asarray(np.diff(timestamps), dtype=np.float32)
        arrays = self._arrays(feeds)
        pointers = (C.c_void_p * len(arrays))(*(a.ctypes.data for a in arrays))
        output = np.empty_like(feeds["x"], dtype=np.float32)
        rc = self.lib.nano_trt_run4(
            self.handle,
            pointers,
            times.ctypes.data,
            delta.ctypes.data,
            output.ctypes.data,
        )
        if rc:
            raise RuntimeError(self.lib.nano_trt_error().decode())
        return output

    @property
    def last_gpu_ms(self):
        return self.lib.nano_trt_last_ms(self.handle)

    def close(self):
        if self.handle:
            self.lib.nano_trt_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class GpuIstft:
    """FP32, centered periodic-Hann ISTFT: n_fft=1024, hop_length=256."""

    def __init__(self, frames, library=DEFAULT_LIBRARY):
        if frames < 2:
            raise ValueError("At least two spectral frames are required")
        self.frames = frames
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.nano_istft_create.argtypes = [C.c_int]
        self.lib.nano_istft_create.restype = C.c_void_p
        self.lib.nano_istft_run.argtypes = [C.c_void_p] * 5
        self.lib.nano_istft_destroy.argtypes = [C.c_void_p]
        self.lib.nano_istft_destroy.restype = None
        self.lib.nano_istft_error.restype = C.c_char_p
        self.handle = self.lib.nano_istft_create(frames)
        if not self.handle:
            raise RuntimeError(self.lib.nano_istft_error().decode())
        self.output = np.empty((frames - 1) * 256, dtype=np.float32)

    def __call__(self, mag, cosine, sine):
        if not self.handle:
            raise RuntimeError("ISTFT has been closed")
        arrays = [
            np.ascontiguousarray(x, dtype=np.float32) for x in (mag, cosine, sine)
        ]
        if any(a.shape != (513, self.frames) for a in arrays):
            raise ValueError("Expected frequency-major [513, frames] spectra")
        rc = self.lib.nano_istft_run(
            self.handle,
            *[C.c_void_p(a.ctypes.data) for a in arrays],
            C.c_void_p(self.output.ctypes.data),
        )
        if rc:
            raise RuntimeError(self.lib.nano_istft_error().decode())
        return self.output

    def close(self):
        if self.handle:
            self.lib.nano_istft_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
