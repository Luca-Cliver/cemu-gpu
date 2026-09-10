"""Compatibility exports for shared layer-weight prefetching."""

from runtime_common.weight_prefetch import ModelWeightPrefetcher, ModelWeightRequest

FlexGenWeightPrefetcher = ModelWeightPrefetcher
FlexGenWeightRequest = ModelWeightRequest

__all__ = ["FlexGenWeightPrefetcher", "FlexGenWeightRequest"]
