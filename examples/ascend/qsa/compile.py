"""Compilation entry point for the static and dynamic-shape QSA kernels."""

import tilelang

from examples.ascend.qsa.kernel import qsa_fwd


def compile_qsa(num_tokens, kv_length, top_k=2176, *, dynamic_shape=False, variant="baseline", pad_m=32):
    if variant == "baseline":
        if pad_m != 32:
            raise ValueError("Baseline QSA requires pad_m=32")
        program = qsa_fwd(num_tokens, kv_length, top_k, dynamic_shape=dynamic_shape)
    elif variant in ("author-cube", "author-aiv"):
        from examples.ascend.qsa.kernel_gm2l1 import qsa_fwd as author_qsa

        program = author_qsa(num_tokens, kv_length, top_k, dynamic_shape=dynamic_shape, cube_gather=variant == "author-cube", pad_m=pad_m)
    else:
        raise ValueError(f"Unknown QSA variant {variant}")
    return tilelang.compile(
        program,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
