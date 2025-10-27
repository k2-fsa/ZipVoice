import torch
import tensorrt as trt
import logging
import os
import queue

class TrtContextWrapper:
    def __init__(self, trt_engine, trt_concurrent=1, device='cuda:0'):
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

    def acquire_estimator(self):
        return self.trt_context_pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self.trt_context_pool.put([context, stream])

    def __call__(self, x, t, padding_mask, guidance_scale):
        x = x.to(torch.float16)
        t = t.to(torch.float16)
        padding_mask = padding_mask.to(torch.float16)
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
            estimator.set_input_shape('guidance_scale', (batch_size,))
            
            # Set input tensor addresses
            input_data_ptrs = [x.contiguous().data_ptr(), t.contiguous().data_ptr(), padding_mask.contiguous().data_ptr(), guidance_scale.contiguous().data_ptr()]
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
        # breakpoint()
        return output.to(torch.float32)



def load_trt(model, trt_model, trt_concurrent=1):
    assert os.path.exists(trt_model), f'Please export trt model first.'
    import tensorrt as trt
    with open(trt_model, 'rb') as f:
        estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
    assert estimator_engine is not None, 'failed to load trt {}'.format(trt_model)
    del model.fm_decoder
    model.fm_decoder = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device='cuda')


