import json
import math
from pathlib import Path


IMPLEMENTATION = "torch.softmax.float32.out.cuda_graph"


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_time(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def sparf_softmax_shapes(batch_sizes, heads, prompt_length, decode_steps, compression_ratio):
    for name, value in (
        ("heads", heads), ("prompt_length", prompt_length),
        ("decode_steps", decode_steps), ("compression_ratio", compression_ratio),
    ):
        positive_integer(value, name)
    if not batch_sizes:
        raise ValueError("batch_sizes must not be empty")
    shapes = set()
    for batch in batch_sizes:
        positive_integer(batch, "batch size")
        for tokens in range(prompt_length + 1, prompt_length + decode_steps + 1):
            selected = (tokens + compression_ratio - 1) // compression_ratio
            shapes.add((batch * heads, tokens))
            shapes.add((batch * heads, selected))
    return sorted(shapes)


class SoftmaxRatioTable:
    def __init__(self, report):
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            raise ValueError("unsupported Softmax ratio report schema")
        if report.get("implementation") != IMPLEMENTATION:
            raise ValueError("unsupported Softmax measurement implementation")
        anchor = report.get("anchor")
        if not isinstance(anchor, dict):
            raise ValueError("missing Softmax anchor")
        self.anchor_vectors = positive_integer(anchor.get("vectors"), "anchor vectors")
        self.anchor_tokens = positive_integer(anchor.get("tokens"), "anchor tokens")
        self.anchor_us = positive_time(anchor.get("median_us"), "anchor median_us")
        measurements = report.get("measurements")
        if not isinstance(measurements, list) or not measurements:
            raise ValueError("missing Softmax measurements")
        self._times = {}
        for measurement in measurements:
            if not isinstance(measurement, dict):
                raise ValueError("Softmax measurement must be an object")
            shape = (
                positive_integer(measurement.get("vectors"), "vectors"),
                positive_integer(measurement.get("tokens"), "tokens"),
            )
            if shape in self._times:
                raise ValueError(f"duplicate Softmax shape {shape}")
            self._times[shape] = positive_time(measurement.get("median_us"), "median_us")
        anchor_shape = (self.anchor_vectors, self.anchor_tokens)
        if self._times.get(anchor_shape) != self.anchor_us:
            raise ValueError("anchor measurement does not match anchor median_us")

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as report_file:
            return cls(json.load(report_file))

    def validate_anchor(self, device):
        expected = (device.softmax_anchor_heads, device.softmax_anchor_tokens)
        if (self.anchor_vectors, self.anchor_tokens) != expected:
            raise ValueError(f"Softmax anchor shape must match runtime config {expected}")

    def ratio(self, vectors, tokens):
        positive_integer(vectors, "vectors")
        positive_integer(tokens, "tokens")
        try:
            measured_us = self._times[(vectors, tokens)]
        except KeyError as error:
            raise ValueError(
                f"unmeasured Softmax shape vectors={vectors}, tokens={tokens}; "
                "measure this GPU microbatch size and Decode length first"
            ) from error
        return measured_us / self.anchor_us

    def estimate_ns(self, device, vectors, tokens):
        self.validate_anchor(device)
        anchor_us = positive_time(device.softmax_anchor_latency_us, "paper anchor latency")
        return math.ceil(anchor_us * 1000 * self.ratio(vectors, tokens))

    def require_sparf(self, batch_size, heads, prompt_length, decode_steps, compression_ratio):
        for vectors, tokens in sparf_softmax_shapes(
            [batch_size], heads, prompt_length, decode_steps, compression_ratio,
        ):
            self.ratio(vectors, tokens)
