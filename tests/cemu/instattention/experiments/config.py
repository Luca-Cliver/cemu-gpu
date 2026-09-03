import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_layers: int
    hidden_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    max_sequence_length: int
    dtype: str


@dataclass(frozen=True)
class WorkloadConfig:
    prompt_length: int
    decode_length: int
    batch_sizes: Tuple[int, ...]
    attention_mode: str


@dataclass(frozen=True)
class InstCsdConfig:
    count: int
    flash_channels: int
    channel_bandwidth_gbps: float
    aggregate_bandwidth_gbps: float
    external_pcie_gbps: float
    gemv_gflops: float
    softmax_mflops: float
    filter_gbps: float
    softmax_anchor_heads: int
    softmax_anchor_tokens: int
    softmax_anchor_latency_us: float


@dataclass(frozen=True)
class MeasurementConfig:
    warmup_iterations: int
    iterations: int


@dataclass(frozen=True)
class InstAttentionExperimentConfig:
    model: ModelConfig
    workload: WorkloadConfig
    instcsd: InstCsdConfig
    measurement: MeasurementConfig


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _positive_int(section: Mapping[str, Any], key: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _non_negative_int(section: Mapping[str, Any], key: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _positive_float(section: Mapping[str, Any], key: str) -> float:
    value = section.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be positive")
    return float(value)


def _text(section: Mapping[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def load_experiment_config(path: Any) -> InstAttentionExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        raw = json.load(config_file)
    if not isinstance(raw, Mapping):
        raise ValueError("experiment configuration must be a JSON object")

    model_raw = _mapping(raw, "model")
    workload_raw = _mapping(raw, "workload")
    instcsd_raw = _mapping(raw, "instcsd")
    measurement_raw = _mapping(raw, "measurement")

    model = ModelConfig(
        name=_text(model_raw, "name"),
        num_layers=_positive_int(model_raw, "num_layers"),
        hidden_size=_positive_int(model_raw, "hidden_size"),
        num_query_heads=_positive_int(model_raw, "num_query_heads"),
        num_kv_heads=_positive_int(model_raw, "num_kv_heads"),
        head_dim=_positive_int(model_raw, "head_dim"),
        intermediate_size=_positive_int(model_raw, "intermediate_size"),
        max_sequence_length=_positive_int(model_raw, "max_sequence_length"),
        dtype=_text(model_raw, "dtype"),
    )
    if model.hidden_size != model.num_query_heads * model.head_dim:
        raise ValueError("hidden_size must equal num_query_heads * head_dim")
    if model.num_query_heads % model.num_kv_heads != 0:
        raise ValueError("num_kv_heads must divide num_query_heads")
    if model.dtype not in ("float16", "float32"):
        raise ValueError("dtype must be float16 or float32")

    batch_sizes_raw = workload_raw.get("batch_sizes")
    if not isinstance(batch_sizes_raw, list) or not batch_sizes_raw:
        raise ValueError("batch_sizes must be a non-empty array")
    batch_sizes = tuple(
        _positive_int({"batch_size": value}, "batch_size")
        for value in batch_sizes_raw
    )
    workload = WorkloadConfig(
        prompt_length=_positive_int(workload_raw, "prompt_length"),
        decode_length=_positive_int(workload_raw, "decode_length"),
        batch_sizes=batch_sizes,
        attention_mode=_text(workload_raw, "attention_mode"),
    )
    if workload.attention_mode != "dense":
        raise ValueError("the initial experiment only supports dense attention")
    if workload.prompt_length + workload.decode_length > model.max_sequence_length:
        raise ValueError("prompt_length + decode_length exceeds max_sequence_length")

    instcsd = InstCsdConfig(
        count=_positive_int(instcsd_raw, "count"),
        flash_channels=_positive_int(instcsd_raw, "flash_channels"),
        channel_bandwidth_gbps=_positive_float(
            instcsd_raw, "channel_bandwidth_gbps"
        ),
        aggregate_bandwidth_gbps=_positive_float(
            instcsd_raw, "aggregate_bandwidth_gbps"
        ),
        external_pcie_gbps=_positive_float(instcsd_raw, "external_pcie_gbps"),
        gemv_gflops=_positive_float(instcsd_raw, "gemv_gflops"),
        softmax_mflops=_positive_float(instcsd_raw, "softmax_mflops"),
        filter_gbps=_positive_float(instcsd_raw, "filter_gbps"),
        softmax_anchor_heads=_positive_int(instcsd_raw, "softmax_anchor_heads"),
        softmax_anchor_tokens=_positive_int(instcsd_raw, "softmax_anchor_tokens"),
        softmax_anchor_latency_us=_positive_float(
            instcsd_raw, "softmax_anchor_latency_us"
        ),
    )

    measurement = MeasurementConfig(
        warmup_iterations=_non_negative_int(
            measurement_raw, "warmup_iterations"
        ),
        iterations=_positive_int(measurement_raw, "iterations"),
    )
    return InstAttentionExperimentConfig(
        model=model,
        workload=workload,
        instcsd=instcsd,
        measurement=measurement,
    )
