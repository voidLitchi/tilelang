"""Compilation entry point for the static and dynamic-shape QSA kernels."""

import tilelang

from examples.ascend.qsa.kernel import qsa_fwd


def compile_qsa(
    num_tokens, kv_length, top_k=2176, *, dynamic_shape=False, variant="baseline", pad_m=32, num_stages=None, aic_count=None
):
    configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    if aic_count is not None and (aic_count not in (28, 32, 36) or variant in ("baseline", "author-cube")):
        raise ValueError("--aic-count requires 28, 32 or 36 and an AIV or hybrid variant")
    if variant == "baseline":
        if pad_m != 32:
            raise ValueError("Baseline QSA requires pad_m=32")
        if num_stages is not None:
            raise ValueError("The --stages override requires an author QSA variant")
        program = qsa_fwd(num_tokens, kv_length, top_k, dynamic_shape=dynamic_shape)
    elif variant in (
        "old-schedule",
        "old-schedule-tail",
        "old-schedule-tail-late",
        "old-prefetch",
        "old-chunk-ahead",
        "old-hybrid-k",
        "old-hybrid-k-ahead",
    ):
        from examples.ascend.qsa.kernel_old_schedule import qsa_old_schedule

        if dynamic_shape and variant not in ("old-schedule", "old-schedule-tail", "old-schedule-tail-late"):
            raise ValueError("Only old-schedule and old-schedule-tail support dynamic shapes")
        if pad_m != 16 or top_k != 2176 or num_stages not in (None, 3):
            raise ValueError("The explicit old schedule requires pad_m=16, 2176 slots and 3 stages")
        program = qsa_old_schedule(
            num_tokens, kv_length, prefetch=variant == "old-prefetch", aic_count=aic_count,
            direct_k=variant in ("old-hybrid-k", "old-hybrid-k-ahead"),
            lookahead_k=variant == "old-hybrid-k-ahead", pair_chunks=variant == "old-chunk-ahead",
            skip_tail=variant in ("old-schedule-tail", "old-schedule-tail-late"),
            tail_sync_late=variant == "old-schedule-tail-late",
            dynamic_shape=dynamic_shape,
        )
        configs[tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE] = False
    elif variant in ("author-cube", "author-aiv", "padded-aiv", "hybrid-k"):
        from examples.ascend.qsa.kernel_gm2l1 import qsa_fwd as author_qsa

        program = author_qsa(
            num_tokens,
            kv_length,
            top_k,
            dynamic_shape=dynamic_shape,
            cube_gather=variant == "author-cube",
            pad_m=pad_m,
            num_stages=num_stages,
            aic_count=aic_count,
            padded_gather=variant in ("padded-aiv", "hybrid-k"),
            direct_k=variant == "hybrid-k",
        )
    else:
        raise ValueError(f"Unknown QSA variant {variant}")
    return tilelang.compile(
        program,
        out_idx=-1,
        pass_configs=configs,
    )
