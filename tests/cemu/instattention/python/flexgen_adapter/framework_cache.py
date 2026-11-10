from concurrent.futures import ThreadPoolExecutor
from collections import deque
from threading import Lock

import numpy as np
import torch

from phase_profiler import profile_scope, stack_profiler, submit_profiled, profile_request


def producer_event(device):
    if device.type != "cuda":
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    return event


class DeviceKvCache:
    def __init__(self, session, layer, microbatch):
        self.session = session
        self.layer = layer
        self.microbatch = microbatch
        self.backend = session.backends[microbatch]
        self.prefill = None
        self.deleted = False

    def submit_prefill(self, keys, values, event):
        if self.prefill is not None:
            raise RuntimeError("Prefill KV was already submitted")

        def write():
            if event is not None:
                with profile_scope(stack_profiler(self.session.profiler), "prefill.producer_wait_wall"):
                    event.synchronize()
            with profile_scope(self.session.profiler, "framework.prefill_kv_worker_wall"):
                return self.backend.write_prefill(self.layer, keys, values)

        self.prefill = self.session.submit_prefill(write)

    def submit_decode(self, token, query, keys, values, event):
        if self.prefill is None:
            raise RuntimeError("Decode requires submitted Prefill KV")

        def execute():
            with profile_scope(stack_profiler(self.session.profiler), "decode.layer_prefill_wait_wall"):
                self.prefill.result()
            if token == self.session.task.prompt_len:
                with profile_scope(self.session.profiler, "framework.prefill_accept_wait_wall"):
                    for cache in self.session.caches:
                        if cache.microbatch == self.microbatch and cache.prefill is not None:
                            cache.prefill.result()
            if event is not None:
                with profile_scope(stack_profiler(self.session.profiler), "decode.producer_wait_wall"):
                    event.synchronize()
            with profile_scope(self.session.profiler, "framework.decode_append_wall"):
                self.backend.append_decode(self.layer, token, keys, values)
            request = self.backend.submit_decode(self.layer, query, token + 1)
            return self.backend.wait_decode(request)

        context = dict(id=f"decode:{self.layer}:{self.microbatch}:{token}",
                       layer=self.layer, microbatch=self.microbatch, tokens=token + 1)
        with profile_request(self.session.profiler, context):
            future = self.session.submit(execute)
        future.cemu_profile_request = context
        return future

    def delete(self):
        if not self.deleted:
            if self.prefill is not None:
                self.prefill.result()
            self.deleted = True


class PendingAttention:
    def __init__(self, session, future, inputs, finish):
        self.session = session
        self.future = future
        self.shape = inputs.shape
        self.dtype = inputs.dtype
        self.device = inputs.device
        self.finish = finish
        self.resolved = None

    @property
    def data(self):
        if self.resolved is None:
            with profile_request(self.session.profiler, getattr(self.future, "cemu_profile_request", None)), \
                    profile_scope(self.session.profiler, "framework.attention_result_wait_wall"):
                output = self.future.result()
            self.resolved = self.finish(output)
            self.finish = None
        return self.resolved.data

    def move(self, device):
        if device != self.device:
            raise RuntimeError("pending CEMU Attention requires GPU-resident activations")
        return self

    def delete(self):
        self.data
        self.resolved.delete()


class FrameworkSession:
    def __init__(self, backends, profiler=None, config=None, policy=None):
        self.backends = tuple(backends)
        if not self.backends:
            raise ValueError("at least one CEMU microbatch backend is required")
        self.profiler = profiler
        self.config = config
        self.policy = policy
        self.prefill_queue = deque()
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flexgen-kv")
        self.attention = ThreadPoolExecutor(max_workers=len(self.backends), thread_name_prefix="flexgen-csd")
        self.task = None
        self.caches = []
        self.futures = set()
        self.lock = Lock()
        self.submitted = 0
        self.peak_pending = 0
        self.failure = None
        self.closed = False

    def start_task(self, task):
        if self.task is not None and self.task is not task:
            raise RuntimeError("create a fresh CEMU session for each generate() call")
        if self.task is None and self.config is not None:
            inputs = np.asarray(task.inputs)
            maximum = task.prompt_len + task.gen_len - 1
            if (task.gen_len < 2 or task.do_sample or task.stop is not None or task.cut_gen_len is not None or
                    inputs.shape != (self.policy.gpu_batch_size * self.policy.num_gpu_batches, task.prompt_len) or
                    not np.issubdtype(inputs.dtype, np.integer) or
                    np.any(inputs < 0) or np.any(inputs >= self.config.vocab_size) or
                    np.any(inputs == self.config.pad_token_id) or maximum > self.config.max_seq_len or
                    any(maximum > backend.layout.config.max_seq_len for backend in self.backends)):
                raise ValueError("CEMU FlexGen requires unpadded, in-capacity prompts and greedy full-length generation with at least one Decode")
        self.task = task

    def submit_prefill(self, function):
        if len(self.prefill_queue) >= 2 * len(self.backends):
            with profile_scope(self.profiler, "prefill.writer_backpressure_wall"):
                self.prefill_queue.popleft().result()
        future = submit_profiled(self.writer.submit, function, self.profiler, "prefill.writer")
        self.prefill_queue.append(future)
        return future

    def cache(self, layer, microbatch):
        cache = DeviceKvCache(self, layer, microbatch)
        self.caches.append(cache)
        return cache

    def submit(self, function):
        with self.lock:
            if self.failure is not None or self.closed:
                raise RuntimeError("CEMU framework session is failed or closed") from self.failure
            future = submit_profiled(self.attention.submit, function, self.profiler, "decode.attention")
            self.futures.add(future)
            self.submitted += 1
            self.peak_pending = max(self.peak_pending, len(self.futures))
        future.add_done_callback(self._complete)
        return future

    def _complete(self, future):
        with self.lock:
            self.futures.discard(future)
            if future.exception() is not None:
                self.failure = future.exception()

    def wait(self):
        for cache in self.caches:
            if cache.prefill is not None:
                cache.prefill.result()
        with self.lock:
            futures = tuple(self.futures)
        for future in futures:
            future.result()
        if self.failure is not None:
            raise RuntimeError("CEMU Attention failed") from self.failure

    def drain(self):
        self.wait()
        stores = {}
        for backend in self.backends:
            stores[id(backend.store)] = backend.store
        for store in stores.values():
            store.drain()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.writer.shutdown(wait=True)
        self.attention.shutdown(wait=True)
