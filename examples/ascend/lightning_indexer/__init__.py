"""DeepSeek-V4.1-Flash Lightning Indexer Sparse Backward operator for Ascend 950."""

from __future__ import annotations

from .reference import (
    SPEC_THRESHOLDS,
    AccuracyMetrics,
    compute_accuracy_metrics,
    ref_lightning_indexer_sparse_bwd,
    ref_lightning_indexer_sparse_bwd_analytical,
    ref_lightning_indexer_sparse_fwd,
)

_KERNEL_EXPORTS = (
    "build_lightning_indexer_sparse_bwd_kernel",
    "get_compile_count",
    "lightning_indexer_sparse_bwd",
    "reset_compile_cache",
)

__all__ = [
    "AccuracyMetrics",
    "SPEC_THRESHOLDS",
    "build_lightning_indexer_sparse_bwd_kernel",
    "compute_accuracy_metrics",
    "get_compile_count",
    "lightning_indexer_sparse_bwd",
    "ref_lightning_indexer_sparse_bwd",
    "ref_lightning_indexer_sparse_bwd_analytical",
    "ref_lightning_indexer_sparse_fwd",
    "reset_compile_cache",
]


def __getattr__(name: str):
    if name in _KERNEL_EXPORTS:
        from . import kernel

        return getattr(kernel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
