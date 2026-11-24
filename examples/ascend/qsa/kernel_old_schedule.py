"""Final QSA schedule for Qwen3.8-Flash-Next on Ascend 950."""

from pathlib import Path

import tilelang.ascend.language as T


NUM_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
GROUP_SIZE = NUM_HEADS // NUM_KV_HEADS
LOGICAL_TOP_K = 2051
PHYSICAL_TOP_K = 2176
PAD_M = 16
PIPELINE_STAGES = 3
SUPPORTED_AIC_COUNTS = (28, 32, 36)


def qsa_old_schedule(total_q, total_kv, *, aic_count=32, dynamic_shape=False):
    """Build the fixed QSA schedule with dynamic packed Q/KV row counts."""
    assert aic_count in SUPPORTED_AIC_COUNTS, "aic_count must be 28, 32, or 36"
    assert total_q > 0 and total_kv >= 4, "Q must be positive and KV must be at least four rows"

    # This schedule passes 32-bit GM offsets to AscendC. Keep the compile-time
    # checks for static builds, and apply the same contract to dynamic launches.
    if not dynamic_shape:
        assert max(total_q * 8192, (total_kv + 3) * 512) <= 2**31, "the explicit schedule requires int32-addressable GM offsets"

    prelude = Path(__file__).with_name("qsa_old_schedule.h").read_text(encoding="utf-8")
    prelude = "#define QSA_SKIP_TAIL\n#define QSA_DELAY_TAIL_SYNC\n" + prelude

    if dynamic_shape:
        # The same 17-tile schedule and persistent block count serve all packed
        # row counts. The host supplies three physical KV padding rows.
        total_q = T.dynamic("num_tokens")
        padded_kv = T.dynamic("padded_kv_length")
        task = f"qsa_old_task<false, -1, {aic_count}, -1>"
        task_args = (padded_kv - 7, total_q * 2)
        grid = aic_count
    else:
        padded_kv = total_kv + 3
        task = f"qsa_old_task<false, {total_kv - 4}, {aic_count}, {total_q * 2}>"
        task_args = (total_kv - 4, total_q * 2)
        grid = aic_count

    ub_bytes = 111104 + 2 * PHYSICAL_TOP_K * 4

    @T.prim_func
    def main(
        Q: T.Buffer((total_q, NUM_KV_HEADS, PAD_M, HEAD_DIM), "bfloat16"),
        K: T.Buffer((padded_kv, NUM_KV_HEADS, HEAD_DIM), "bfloat16"),
        V: T.Buffer((padded_kv, NUM_KV_HEADS, HEAD_DIM), "bfloat16"),
        Indices: T.Buffer((total_q, PHYSICAL_TOP_K), "int32"),
        Valid: T.Buffer((total_q, PHYSICAL_TOP_K), "int32"),
        O: T.Buffer((total_q, NUM_KV_HEADS, PAD_M, HEAD_DIM), "bfloat16"),
    ):
        with T.MixedKernel(grid, prelude=prelude) as (core, sid):
            scratch = T.alloc_shared((ub_bytes,), "uint8")
            with T.Cube():
                T.evaluate(
                    T.call_extern(
                        "void",
                        task,
                        T.access_ptr(scratch[0], "rw"),
                        Indices.data,
                        K.data,
                        O.data,
                        Q.data,
                        V.data,
                        Valid.data,
                        *task_args,
                    )
                )
            with T.Vector():
                T.evaluate(
                    T.call_extern(
                        "void",
                        task,
                        T.access_ptr(scratch[0], "rw"),
                        Indices.data,
                        K.data,
                        O.data,
                        Q.data,
                        V.data,
                        Valid.data,
                        *task_args,
                    )
                )

    return main
