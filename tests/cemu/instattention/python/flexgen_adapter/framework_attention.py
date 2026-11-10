import torch
from phase_profiler import profile_tensor_scope, stack_profiler

from .framework_import import load_framework
from .framework_cache import PendingAttention, producer_event


flexgen = load_framework()


class CemuSelfAttention(flexgen.SelfAttention):
    def __init__(self, config, env, policy, layer_id, session):
        super().__init__(config, env, policy, layer_id)
        self.session = session
        self.cache_index = 0

    def set_task(self, task):
        super().set_task(task)
        self.session.start_task(task)
        self.cache_index = 0

    def init_cache_one_gpu_batch(self, cache_home):
        cache = self.session.cache(self.layer_id, self.cache_index)
        cache_home.store((cache, cache))
        self.cache_index += 1

    def load_cache(self, cache_home, cache_read_buf, step):
        if step:
            cache_read_buf.store(cache_home.val[0])

    def store_cache(self, cache_home, cache_write_buf, step):
        pending = cache_write_buf.pop()
        if step == 0:
            cache_home.val[0].submit_prefill(*pending)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, step, microbatch):
        if step == 0:
            with profile_tensor_scope(stack_profiler(self.session.profiler), "prefill.gpu_attention", self.compute.dev):
                super().forward(hidden, cache_read_buf, weight_read_buf, attention_mask,
                                cache_write_buf, step, microbatch)
            keys, values = cache_write_buf.pop()
            shape = (keys.shape[0], self.policy.gpu_batch_size, self.config.n_head,
                     self.config.input_dim // self.config.n_head)
            cache_write_buf.store((keys.data.detach().reshape(shape),
                                   values.data.detach().reshape(shape), producer_event(self.compute.dev)))
            return
        loaded = (weight_read_buf.pop() if microbatch == self.policy.num_gpu_batches - 1
                  else weight_read_buf.val)
        weights = tuple(item[0] for item in loaded)
        inputs = hidden.val
        with profile_tensor_scope(stack_profiler(self.session.profiler), "decode.qkv", self.compute.dev):
            query, keys, values = self.compute.opt_project(inputs, weights, self.config.n_head)
        cache = cache_read_buf.pop()
        future = cache.submit_decode(self.task.prompt_len + step - 1, query[:, :, 0].detach(),
                                     keys.detach(), values.detach(), producer_event(self.compute.dev))

        def finish(output):
            with profile_tensor_scope(stack_profiler(self.session.profiler), "decode.output_to_gpu", self.compute.dev):
                tensor = torch.as_tensor(output).to(device=self.compute.dev, dtype=inputs.dtype)
            expected = (self.policy.gpu_batch_size, self.config.n_head,
                        self.config.input_dim // self.config.n_head)
            if tuple(tensor.shape) != expected:
                raise RuntimeError(f"CEMU Attention output shape {tuple(tensor.shape)} != {expected}")
            with profile_tensor_scope(stack_profiler(self.session.profiler), "decode.wo_residual", self.compute.dev):
                return self.compute.opt_finish(inputs, tensor.unsqueeze(2), weights[6], weights[7])

        hidden.val = PendingAttention(self.session, future, inputs, finish)
        cache_write_buf.store(())
