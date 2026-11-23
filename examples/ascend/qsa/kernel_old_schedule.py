"""Explicit qsa-old AIV schedule for the 2176-slot performance experiment."""

from pathlib import Path

import tilelang.ascend.language as T


def qsa_old_schedule(
    total_q,
    total_kv,
    prefetch,
    aic_count,
    *,
    direct_k=False,
    lookahead_k=False,
    pair_chunks=False,
    skip_tail=False,
    dynamic_shape=False,
):
    if direct_k and prefetch:
        raise ValueError("Direct K uses the tile-wise index ring for V")
    if lookahead_k and not direct_k:
        raise ValueError("K lookahead requires direct K")
    # The captured schedule indexes GM in int32 elements.
    if not dynamic_shape and (min(total_q, total_kv) < 1 or max(total_q * 8192, (total_kv + 3) * 512) > 2**31):
        raise ValueError("The explicit schedule requires nonempty inputs addressable with int32 element offsets")
    if dynamic_shape and aic_count is None:
        raise ValueError("The dynamic explicit schedule requires a fixed persistent block count")
    prelude = Path(__file__).with_name("qsa_old_schedule.h").read_text(encoding="utf-8")
    if direct_k:
        k_load = Path(__file__).with_name("qsa_k_l1.h").read_text(encoding="utf-8")
        prelude = f"#define QSA_K_AHEAD {int(lookahead_k)}\n{k_load}\n{prelude}"
    if pair_chunks:
        prelude = "#define QSA_PAIR_CHUNKS\n" + prelude
    if skip_tail:
        prelude = "#define QSA_SKIP_TAIL\n" + prelude
    if dynamic_shape:
        # Keep the 17-tile schedule and persistent block count fixed. Only
        # the packed task count and the last full KV quad are runtime values.
        total_q = T.dynamic("num_tokens")
        padded_kv = T.dynamic("padded_kv_length")
        task = f"qsa_old_task<{str(prefetch).lower()}, -1, {aic_count}, -1>"
        task_args = (padded_kv - 7, total_q * 2)
        grid = aic_count
    else:
        padded_kv = total_kv + 3
        task = f"qsa_old_task<{str(prefetch).lower()}, {total_kv - 4}, {aic_count or 0}, {total_q * 2}>"
        task_args = (total_kv - 4, total_q * 2)
        grid = aic_count or total_q * 2
    index_rows = 2 if aic_count else 1
    ub_bytes = 111104 + (index_rows * 2176 * 4 if prefetch else 0)

    @T.prim_func
    def main(
        Q: T.Buffer((total_q, 2, 16, 256), "bfloat16"),
        K: T.Buffer((padded_kv, 2, 256), "bfloat16"),
        V: T.Buffer((padded_kv, 2, 256), "bfloat16"),
        Indices: T.Buffer((total_q, 2176), "int32"),
        Valid: T.Buffer((total_q, 2176), "int32"),
        O: T.Buffer((total_q, 2, 16, 256), "bfloat16"),
    ):
        with T.MixedKernel(grid, prelude=prelude) as (core, sid):
            scratch = T.alloc_shared((ub_bytes,), "uint8")
            with T.Cube():
                T.evaluate(
                    T.call_extern("void", task, T.access_ptr(scratch[0], "rw"), Indices.data, K.data, O.data, Q.data, V.data, Valid.data, *task_args)
                )
            with T.Vector():
                T.evaluate(
                    T.call_extern("void", task, T.access_ptr(scratch[0], "rw"), Indices.data, K.data, O.data, Q.data, V.data, Valid.data, *task_args)
                )

    return main
