from flex_llama import LlamaLM, Policy
import torch
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
from flexgen.llama_config import InstLlamaConfig, Instget_llama_config, Instdownload_llama_weights
from flexgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)
from flexgen.compression import CompressionConfig

if __name__ == "__main__":
    model_path = "/home/pxr/Llama-2-7b-hf"
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk("/home/pxr/offload_dir/")
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    llama_config = Instget_llama_config(model_path)
    compress_config = CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False)
    policy = Policy(1, 1, 100, 0, 100, 0, 100, 0, True, True, False, False, 1, False, compress_config, False, compress_config)
    
    path = "/home/pxr/llama_weights"
    
    model = LlamaLM(llama_config, env, path, policy)
