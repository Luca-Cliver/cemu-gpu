from .framework_import import load_framework
from .framework_cache import PendingAttention
from phase_profiler import profile_tensor_scope, stack_profiler


class NoDisk:
    def synchronize(self):
        pass

    def allocate(self, *args, **kwargs):
        raise ValueError("disk offload is not enabled for this CEMU baseline")

    def close_copy_threads(self):
        pass


def make_environment(device="cuda:0", layer_norm_eps=1e-5, profiler=None):
    load_framework()
    from flexgen.opt_backend import OptTorchDevice
    from flexgen.utils import ExecutionEnv

    class ProfiledDevice(OptTorchDevice):
        def mlp(self, inputs, *args, **kwargs):
            if isinstance(inputs, PendingAttention):
                inputs.data
            with profile_tensor_scope(profiler, "framework.mlp", self.dev):
                return super().mlp(inputs, *args, **kwargs)

        def opt_output_embed(self, *args, **kwargs):
            with profile_tensor_scope(profiler, "framework.lm_head", self.dev):
                return super().opt_output_embed(*args, **kwargs)

    device_type = ProfiledDevice if stack_profiler(profiler) is not None else OptTorchDevice
    compute = device_type(device, pin_memory=device != "cpu", layer_norm_eps=layer_norm_eps)
    cpu = OptTorchDevice("cpu", pin_memory=device != "cpu", layer_norm_eps=layer_norm_eps)
    return ExecutionEnv(gpu=compute, cpu=cpu, disk=NoDisk())


def make_policy(gpu_batch_size, num_gpu_batches, overlap=True):
    framework = load_framework()
    return framework.Policy(
        gpu_batch_size=gpu_batch_size, num_gpu_batches=num_gpu_batches,
        w_gpu_percent=100, w_cpu_percent=0, cache_gpu_percent=100, cache_cpu_percent=0,
        act_gpu_percent=100, act_cpu_percent=0, overlap=overlap, sep_layer=True,
        pin_weight=False, cpu_cache_compute=False, attn_sparsity=1,
        compress_weight=False, comp_weight_config=None, compress_cache=False, comp_cache_config=None)
