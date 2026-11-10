"""Use vLLM's Worker/runner extension points without patching its scheduler."""

import os
import time

import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from .session import Config, Session
from .model_config import model_geometry
from .profiling import batch_phase, timing_report
from phase_profiler import PhaseProfiler, profile_scope

_runtime = None


def runtime():
    if _runtime is None:
        raise RuntimeError("CEMU backend requires worker_cls=vllm_adapter.worker.CemuWorker")
    return _runtime


class Runtime:
    def __init__(self, vllm_config):
        c = vllm_config
        m, p, s = c.model_config, c.parallel_config, c.scheduler_config
        unsupported = {
            "unsupported model": m.hf_text_config.model_type not in ("llama", "opt"),
            "non-FP16": m.dtype != torch.float16,
            "non-eager execution": not m.enforce_eager,
            "prefix caching": c.cache_config.enable_prefix_caching,
            "chunked prefill": s.enable_chunked_prefill,
            "async scheduling": s.async_scheduling,
            "parallel execution": any(getattr(p, name) != 1 for name in (
                "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                "decode_context_parallel_size", "prefill_context_parallel_size")),
            "dual batch overlap": p.enable_dbo,
            "speculative decoding": c.speculative_config is not None,
            "KV transfer": c.kv_transfer_config is not None,
            "LoRA": c.lora_config is not None,
            "quantization": m.quantization is not None,
            "sleep mode": m.enable_sleep_mode,
            "other Attention backend": c.attention_config.backend.name != "CEMU",
        }
        errors = [name for name, enabled in unsupported.items() if enabled]
        if errors:
            raise ValueError("unsupported CEMU configuration: " + ", ".join(errors))
        cfg = Config.load(os.environ["CEMU_VLLM_CONFIG"])
        if m.max_model_len > cfg.max_tokens or s.max_num_seqs > cfg.request_capacity:
            raise ValueError("vLLM model/batch limits exceed CEMU capacity")
        self.session = Session(cfg, *model_geometry(m.hf_text_config))
        self.layer_ids = {}
        self.max_batch_tokens = s.max_num_batched_tokens
        self.max_requests = s.max_num_seqs
        self.active = False
        self.work = None
        self.staging = None


class ProfiledModelRunner(GPUModelRunner):
    """Measure synchronous steps without adding a CUDA synchronization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode = os.environ.get("CEMU_VLLM_PROFILE", "off")
        self.cemu_profiler = None if mode == "off" else PhaseProfiler(stack=mode == "detailed")
        self._timing_started = None
        self._timing_phase = "empty"

    def _build_attention_metadata(self, *args, **kwargs):
        if self._timing_started is not None and kwargs.get("num_scheduled_tokens") is not None:
            n = len(self.input_batch.req_ids)
            self._timing_phase = batch_phase(
                self.input_batch.num_computed_tokens_cpu[i] < self.input_batch.num_prompt_tokens[i]
                for i in range(n))
        return super()._build_attention_metadata(*args, **kwargs)

    def execute_model(self, *args, **kwargs):
        if self.cemu_profiler is None:
            return super().execute_model(*args, **kwargs)
        if self._timing_started is not None:
            raise RuntimeError("profiling requires synchronous execute/sample steps")
        self._timing_phase = "empty"
        self._timing_started = time.perf_counter_ns()
        try:
            result = super().execute_model(*args, **kwargs)
            self.cemu_profiler.record(self._timing_phase + ".execute_host_wall",
                                      time.perf_counter_ns() - self._timing_started)
            if result is not None:  # Empty step or an already completed output.
                self._finish_timing()
            return result
        except BaseException:
            self._timing_started = None
            raise

    def _finish_timing(self):
        if self._timing_started is not None:
            finished = time.perf_counter_ns()
            self.cemu_profiler.record(self._timing_phase + ".step_wall",
                                      finished - self._timing_started)
            self.cemu_profiler.trace_interval("vllm." + self._timing_phase + ".step_wall",
                                             self._timing_started, finished)
            self._timing_started = None

    def sample_tokens(self, *args, **kwargs):
        if self.cemu_profiler is None or self._timing_started is None:
            return super().sample_tokens(*args, **kwargs)
        try:
            with profile_scope(self.cemu_profiler, self._timing_phase + ".sample_host_wall"):
                result = super().sample_tokens(*args, **kwargs)
            self._finish_timing()
            return result
        except BaseException:
            self._timing_started = None
            raise


class ProfiledWorker(Worker):
    def _make_model_runner(self):
        if self.use_v2_model_runner or self.scheduler_config.async_scheduling:
            raise ValueError("timing requires the synchronous V1 model runner")
        return ProfiledModelRunner(self.vllm_config, self.device)

    def cemu_timing(self):
        profiler = self.model_runner.cemu_profiler
        report = timing_report(profiler)
        if profiler is not None:
            profiler.write_intervals()
        return report


class CemuModelRunner(ProfiledModelRunner):
    def _update_states(self, scheduler_output):
        r = runtime()
        if scheduler_output.preempted_req_ids or scheduler_output.scheduled_cached_reqs.resumed_req_ids:
            raise RuntimeError("CEMU does not support KV preemption/recompute yet")
        r.session.finish(scheduler_output.finished_req_ids)
        return super()._update_states(scheduler_output)

    def _build_attention_metadata(self, *args, **kwargs):
        r = runtime()
        scheduled = kwargs.get("num_scheduled_tokens")
        if r.active and scheduled is not None:
            # Read identities only after vLLM has condensed/reordered input_batch.
            ids = self.input_batch.req_ids
            r.work = r.session.plan(
                ids, self.input_batch.num_computed_tokens_cpu[:len(ids)],
                self.input_batch.num_prompt_tokens[:len(ids)],
                [scheduled[request_id] for request_id in ids])
        else:
            r.work = None
        try:
            return super()._build_attention_metadata(*args, **kwargs)
        finally:
            r.work = None

    def execute_model(self, *args, **kwargs):
        r = runtime()
        r.active = True
        try:
            return super().execute_model(*args, **kwargs)
        except BaseException as error:
            r.session.failure = error
            raise
        finally:
            r.active = False


class CemuWorker(ProfiledWorker):
    def _make_model_runner(self):
        global _runtime
        if self.use_v2_model_runner:
            raise ValueError("CEMU requires VLLM_USE_V2_MODEL_RUNNER=0")
        if _runtime is not None and not _runtime.session.closed:
            raise RuntimeError("only one CEMU worker/session per process is supported")
        _runtime = Runtime(self.vllm_config)
        runner = CemuModelRunner(self.vllm_config, self.device)
        _runtime.session.profiler = runner.cemu_profiler
        return runner

    def cemu_drain(self):
        session = runtime().session
        exec_count = session.device.decode_count if session.device is not None else 0
        inner = session.device._device if session.device is not None else None
        program = inner.program_id if inner is not None else None
        with profile_scope(session.profiler, "terminal_drain.wall"):
            session.close()
        return {"closed": True, "attention_exec_count": exec_count,
                "statistics": session.final_statistics, "program_id": program,
                "timing": self.cemu_timing()}

    def shutdown(self):
        try:
            if _runtime is not None:
                _runtime.session.close()
        finally:
            super().shutdown()
