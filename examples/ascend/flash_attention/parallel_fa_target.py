"""Target-DSL FlashAttention forward: one BR row loop, row-slice T.reduce.

This is the performance-oriented DSL shape for the P1/P2/P3 closed loop. It
keeps the no-fragment/UB-resident state of the current parallel FA, but writes
the softmax as a single outer ``T.Parallel(BR)`` row loop. Row max/sum are
expressed as row-local ``T.reduce_*`` calls on ``S_ub[i, 0:BC]`` /
``P_ub[i, 0:BC]``, so a future P3 emitter can lower them to the same vector
row-reduction instructions used by ``core.py`` without creating separate BR
loops.

P1 support status (as of P1 Phase 0 F4):
  * Supported: the outer row ``T.Parallel`` becomes one region (collapsed axes
    plus the ordered ops inside it), no-fragment shared/UB state, and the
    layout-aware ``P_nz`` store. Statements outside a Parallel loop (top-level
    ``T.copy``/``T.reduce``) are outside P1's unit and are not modelled.
  * NOT supported yet: ``T.copy``/``T.reduce_*`` nested inside ``T.Parallel``.
    The nest is rejected ("opaque Evaluate inside T.Parallel"), so this kernel's
    row-loop region is OPAQUE with reason code UNSUPPORTED. P1 has no op model
    for nested reductions yet; the region tree (nested Parallel plus
    reduce/copy ops) is the next step. This file is a target DSL/reference, not
    a currently liftable kernel.
  * NOT implemented yet in P2/P3: lowering row-slice reduce inside the BR loop
    to vector row reduction, and emitting the combined row loop.

The intended P3 output is one outer BR loop containing:
  save old max -> scale S -> row max -> softmax/P_nz -> row sum -> alpha/l.
"""

import math

import tilelang.ascend.language as T
from tilelang.layout import make_ascend_compact_nz_layout

PARALLEL_FA_BR = 128
PARALLEL_FA_BC = 128
PARALLEL_FA_D = 64


def parallel_flash_attention_fwd_target(
        q_len=PARALLEL_FA_BR,
        kv_len=4 * PARALLEL_FA_BC,
        head_dim=PARALLEL_FA_D,
        *,
        dtype="bfloat16",
        accum_dtype="float32",
        softmax_scale=None,
):
    """Build the target-DSL kernel (not yet P1-liftable)."""
    BR = q_len
    D = head_dim
    BC_TILE = PARALLEL_FA_BC
    assert BR % 16 == 0
    assert kv_len % BC_TILE == 0
    assert D % 16 == 0
    NUM_KV = kv_len // BC_TILE
    NUM_STAGES = 3
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)

    @T.prim_func
    def main(
            Q: T.Tensor((BR, D), dtype),
            K: T.Tensor((kv_len, D), dtype),
            V: T.Tensor((kv_len, D), dtype),
            O: T.Tensor((BR, D), accum_dtype),
    ):
        with T.Kernel(1):
            Q_shared = T.alloc_l1((BR, D), dtype)
            K_shared = T.alloc_l1((BC_TILE, D), dtype)
            V_shared = T.alloc_l1((D, BC_TILE), dtype)
            P_shared = T.alloc_l1((BR, BC_TILE), dtype)

            qk_a_l0 = T.alloc_l0a((BR, D), dtype)
            qk_b_l0 = T.alloc_l0b((BC_TILE, D), dtype)
            pv_a_l0 = T.alloc_l0a((BR, BC_TILE), dtype)
            pv_b_l0 = T.alloc_l0b((D, BC_TILE), dtype)
            qk_acc_l0c = T.alloc_l0c((BR, BC_TILE), accum_dtype)
            pv_acc_l0c = T.alloc_l0c((BR, D), accum_dtype)

            S_ub = T.alloc_shared((BR, BC_TILE), accum_dtype)
            P_ub = T.alloc_shared((BR, BC_TILE), accum_dtype)
            P_nz_ub = T.alloc_shared((BR + 1, BC_TILE), dtype)
            T.annotate_layout({P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)})
            O_tmp_ub = T.alloc_shared((BR, D), accum_dtype)
            O_ub = T.alloc_shared((BR, D), accum_dtype)

            m_old = T.alloc_shared((BR,), accum_dtype)
            m_new = T.alloc_shared((BR,), accum_dtype)
            l_old = T.alloc_shared((BR,), accum_dtype)
            l_cur = T.alloc_shared((BR,), accum_dtype)
            alpha = T.alloc_shared((BR,), accum_dtype)

            T.copy(Q, Q_shared)

            # Online-softmax state. m_new starts at the running-max sentinel;
            # the row reduce below accumulates into it with clear=False.
            with T.SimdVF():
                for i in T.Parallel(BR):
                    m_old[i] = -T.float32(3.0e38)
                    m_new[i] = -T.float32(3.0e38)
                    l_old[i] = T.float32(0.0)
                    l_cur[i] = T.float32(0.0)
            with T.SimdVF():
                for i, j in T.Parallel(BR, D):
                    O_ub[i, j] = T.float32(0.0)

            for kv in T.Pipelined(NUM_KV, num_stages=NUM_STAGES):
                T.copy(K[kv * BC_TILE : (kv + 1) * BC_TILE, 0:D], K_shared)
                T.copy(V[kv * BC_TILE : (kv + 1) * BC_TILE, 0:D], V_shared, transpose=True)

                # QK^T
                T.copy(Q_shared, qk_a_l0)
                T.copy(K_shared, qk_b_l0)
                T.gemm(qk_a_l0, qk_b_l0, qk_acc_l0c, transpose_B=True, clear_accum=True)
                T.copy(qk_acc_l0c, S_ub)

                # One BR row loop, matching the target core.py structure.
                with T.SimdVF():
                    for i in T.Parallel(BR):
                        m_old[i] = m_new[i]

                        for j in T.Parallel(BC_TILE):
                            S_ub[i, j] = S_ub[i, j] * scale

                        # P1 TARGET GAP: row-slice reduce nested in Parallel.
                        # Whole-buffer T.reduce_max(S_ub, m_new, ...) is not
                        # equivalent: it reduces all rows outside this loop.
                        # The row slice is [1, BC_TILE]; dim=1 gives the row max.
                        T.reduce_max(
                            S_ub[i, 0:BC_TILE],
                            m_new[i : i + 1],
                            dim=1,
                            clear=False,
                        )

                        for j in T.Parallel(BC_TILE):
                            P_ub[i, j] = T.exp(S_ub[i, j] - m_new[i])
                            P_nz_ub[i, j] = T.cast(P_ub[i, j], dtype)

                        # P1 TARGET GAP: same nested row-reduction issue.
                        T.reduce_sum(
                            P_ub[i, 0:BC_TILE],
                            l_cur[i : i + 1],
                            dim=1,
                            clear=True,
                        )

                        alpha[i] = T.exp(m_old[i] - m_new[i])
                        l_old[i] = l_old[i] * alpha[i] + l_cur[i]

                # Feed the NZ probabilities to the second GEMM.
                T.copy(P_nz_ub[0:BR, 0:BC_TILE], P_shared[0:BR, 0:BC_TILE])
                T.copy(P_shared, pv_a_l0)
                T.copy(V_shared, pv_b_l0)
                T.gemm(pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_B=True, clear_accum=True)
                T.copy(pv_acc_l0c, O_tmp_ub)

                # O_new = O_old * alpha + P @ V
                with T.SimdVF():
                    for i, j in T.Parallel(BR, D):
                        O_ub[i, j] = O_ub[i, j] * alpha[i] + O_tmp_ub[i, j]

            # Normalize by the final row sum.
            with T.SimdVF():
                for i, j in T.Parallel(BR, D):
                    O_ub[i, j] = O_ub[i, j] / l_old[i]
            T.copy(O_ub, O)

    return main
