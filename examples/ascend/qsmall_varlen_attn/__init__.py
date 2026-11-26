"""Ascend NPU TileLang implementation of qsmall_varlen_attn_fwd."""

from .kernel import (
    DEFAULT_BLOCK_N,
    DEFAULT_H_PACK,
    DEFAULT_L2_CACHE_CTRL,
    DEFAULT_NUM_STAGES,
    build_qsmall_varlen_attn_fwd,
    clear_kernel_cache,
    detect_num_aiv,
    get_compile_count,
    get_compiled_kernel,
    qsmall_varlen_attn_fwd,
)

__all__ = [
    "DEFAULT_BLOCK_N",
    "DEFAULT_H_PACK",
    "DEFAULT_L2_CACHE_CTRL",
    "DEFAULT_NUM_STAGES",
    "build_qsmall_varlen_attn_fwd",
    "clear_kernel_cache",
    "detect_num_aiv",
    "get_compile_count",
    "get_compiled_kernel",
    "qsmall_varlen_attn_fwd",
]
