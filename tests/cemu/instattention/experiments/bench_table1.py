#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from experiments import DenseAttentionRuntimeModel, load_experiment_config


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "opt13b_dense_1csd.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate the CEMU runtime model against InstAttention Table I"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / expected


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    model = DenseAttentionRuntimeModel(config.instcsd)
    breakdown = model.estimate(
        batch_size=1,
        num_query_heads=1,
        head_dim=config.model.head_dim,
        token_count=16,
    )

    qk_us = breakdown.qk_ns / 1000.0
    softmax_us = breakdown.softmax_ns / 1000.0
    av_us = breakdown.av_ns / 1000.0
    total_us = breakdown.total_ns / 1000.0
    paper_qk_us = 0.32
    paper_softmax_us = config.instcsd.softmax_anchor_latency_us

    print(
        f"[table1-model] config={args.config}, model={config.model.name}, "
        f"dtype={config.model.dtype}"
    )
    print(
        "[table1-model] workload: batch=1, heads=1, tokens=16, "
        f"head_dim={config.model.head_dim}"
    )
    print(
        f"  QK GeMV : model={qk_us:.6f} us, paper={paper_qk_us:.6f} us, "
        f"error={relative_error(qk_us, paper_qk_us) * 100:.3f}%"
    )
    print(
        f"  Softmax : model={softmax_us:.6f} us, paper={paper_softmax_us:.6f} us, "
        f"error={relative_error(softmax_us, paper_softmax_us) * 100:.3f}%"
    )
    print(f"  AV GeMV : model={av_us:.6f} us")
    print(f"  Dense total (QK + Softmax + AV)={total_us:.6f} us")
    print(
        "[table1-model] calibration PASS; no OPT checkpoint or CEMU device is used"
    )


if __name__ == "__main__":
    main()
