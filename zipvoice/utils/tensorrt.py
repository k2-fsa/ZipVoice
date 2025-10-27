# Copyright         2025  Nvidia Corp.        (authors: Yuekai Zhang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script provides utility functions for working with TensorRT in ZipVoice.
"""

import logging
import os
import queue
from typing import Any, Tuple, Optional

import torch
import torch.nn as nn


class TrtContextWrapper:
    """A wrapper class for managing TensorRT execution contexts."""

    def __init__(
        self, trt_engine: Any, trt_concurrent: int = 1, device: str = "cuda:0"
    ):
        """
        Initializes the TrtContextWrapper.

        Args:
            trt_engine (Any): The TensorRT engine.
            trt_concurrent (int, optional): The number of concurrent contexts. Defaults to 1.
            device (str, optional): The device to use. Defaults to 'cuda:0'.
        """
        self.trt_context_pool = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        self.device = device
        for _ in range(trt_concurrent):
            trt_context = trt_engine.create_execution_context()
            trt_stream = torch.cuda.stream(torch.cuda.Stream(torch.device(device)))
            assert trt_context is not None, 'failed to create trt context, maybe not enough CUDA memory, try reduce current trt concurrent {}'.format(trt_concurrent)
            self.trt_context_pool.put([trt_context, trt_stream])
        assert self.trt_context_pool.empty() is False, 'no avaialbe estimator context'
        self.feat_dim = 100

    def acquire_estimator(self) -> Tuple[list, Any]:
        """Acquires a TensorRT context from the pool."""
        return self.trt_context_pool.get(), self.trt_engine

    def release_estimator(self, context: Any, stream: Any):
        """
        Releases a TensorRT context back to the pool.

        Args:
            context (Any): The TensorRT context.
            stream (Any): The CUDA stream.
        """
        self.trt_context_pool.put([context, stream])

    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        padding_mask: torch.Tensor,
        guidance_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Executes the TensorRT engine.

        Args:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The time tensor.
            padding_mask (torch.Tensor): The padding mask tensor.
            guidance_scale (torch.Tensor): The guidance scale tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x.to(torch.float16)
        t = t.to(torch.float16)
        padding_mask = padding_mask.to(torch.float16)
        if guidance_scale is not None:
            guidance_scale = guidance_scale.to(torch.float16)
        [estimator, stream], trt_engine = self.acquire_estimator()
        # NOTE need to synchronize when switching stream
        torch.cuda.current_stream().synchronize()
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Create output tensor with shape (N, T, 100)
        output = torch.empty(batch_size, seq_len, self.feat_dim, dtype=x.dtype, device=x.device)
        
        with stream:
            estimator.set_input_shape('x', (batch_size, x.size(1), x.size(2)))
            estimator.set_input_shape('t', (batch_size,))
            estimator.set_input_shape('padding_mask', (batch_size, padding_mask.size(1)))
            if guidance_scale is not None:
                estimator.set_input_shape('guidance_scale', (batch_size,))
            
            # Set input tensor addresses
            input_data_ptrs = [x.contiguous().data_ptr(), t.contiguous().data_ptr(), padding_mask.contiguous().data_ptr()]
            if guidance_scale is not None:
                input_data_ptrs.append(guidance_scale.contiguous().data_ptr())
            for i, j in enumerate(input_data_ptrs):
                estimator.set_tensor_address(trt_engine.get_tensor_name(i), j)
            
            # Set output tensor address
            # The output tensor name should be the last tensor name in the engine
            num_tensors = trt_engine.num_io_tensors
            output_tensor_name = trt_engine.get_tensor_name(num_tensors - 1)  # Last tensor is output
            estimator.set_tensor_address(output_tensor_name, output.contiguous().data_ptr())
            
            # run trt engine
            assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
            torch.cuda.current_stream().synchronize()
        self.release_estimator(estimator, stream)
        return output.to(torch.float32)

def load_trt(model: nn.Module, trt_model: str, trt_concurrent: int = 1):
    """
    Loads a TensorRT engine and replaces the model's fm_decoder with a TrtContextWrapper.

    Args:
        model (nn.Module): The model to modify.
        trt_model (str): The path to the TensorRT engine file.
        trt_concurrent (int, optional): The number of concurrent contexts. Defaults to 1.
    """
    assert os.path.exists(trt_model), f"Please export trt model first."
    import tensorrt as trt
    with open(trt_model, 'rb') as f:
        estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
    assert estimator_engine is not None, 'failed to load trt {}'.format(trt_model)
    del model.fm_decoder
    model.fm_decoder = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device='cuda')
