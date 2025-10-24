import torch
import tensorrt as trt
import logging
import os
import queue

def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, dtype=torch.float16):
    import tensorrt as trt
    logging.info("Converting onnx to trt...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    if dtype == torch.float16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('failed to parse {}'.format(onnx_model))
    # set input shapes
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(trt_kwargs['input_names'][i], trt_kwargs['min_shape'][i], trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    if dtype == torch.float16:
        tensor_dtype = trt.DataType.HALF
    elif dtype == torch.bfloat16:
        tensor_dtype = trt.DataType.BF16
    elif dtype == torch.float32:
        tensor_dtype = trt.DataType.FLOAT
    else:
        raise ValueError('invalid dtype {}'.format(dtype))
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Succesfully convert onnx to trt...")


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

    def acquire_estimator(self):
        return self.trt_context_pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self.trt_context_pool.put([context, stream])


def get_trt_kwargs_dynamic_batch(min_batch_size=1, opt_batch_size=2, max_batch_size=16):
    feat_dim = 300
    min_seq_len = 100
    opt_seq_len = 200
    max_seq_len = 3000
    min_shape = [(min_batch_size, min_seq_len, feat_dim), (min_batch_size,), (min_batch_size, min_seq_len), (min_batch_size,)]
    opt_shape = [(opt_batch_size, opt_seq_len, feat_dim), (opt_batch_size,), (opt_batch_size, opt_seq_len), (opt_batch_size,)]
    max_shape = [(max_batch_size, max_seq_len, feat_dim), (max_batch_size,), (max_batch_size, max_seq_len), (max_batch_size,)]
    input_names = ["x", "t", "padding_mask", "guidance_scale"]
    return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

def load_trt(model, trt_model, onnx_model, trt_concurrent=1, dtype=torch.float16):
    assert os.path.exists(onnx_model), f'Please use tools/export_onnx.py or tools/export_onnx_streaming.py to export onnx model for token2wav first.'
    if not os.path.exists(trt_model) or os.path.getsize(trt_model) == 0:
        trt_kwargs = get_trt_kwargs_dynamic_batch()
        convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, dtype)
    del model.fm_decoder
    import tensorrt as trt
    with open(trt_model, 'rb') as f:
        estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
    assert estimator_engine is not None, 'failed to load trt {}'.format(trt_model)

    model.fm_decoder = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device='cuda')
