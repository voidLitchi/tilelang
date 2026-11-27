"""TileLang Ascend 950 (dav-3510) Lightning Indexer Sparse Backward kernel.

Implements Operator 1 (``lightning_indexer_sparse_bwd``) per
``lightning_indexer_bwd_spec.md`` v1.0 for DeepSeek-V4.1-Flash
(static ``H=32, D=128, K=512``, dynamic ``B, S_q, S_k``, runtime ``scale``):

- **Multi-SKU Core Compatibility**: Supports ``num_cores`` in ``{28, 32, 36}``
  (default ``32`` AIC / 64 AIV for Ascend 950DT acceptance standard; also
  supports 28 AIC / 56 AIV test machines and 36 AIC / 72 AIV full-chip runs).
- **Zero SIMT**: All dense vector math (scaling, ReLU mask, invalid-slot masking,
  ``dW`` cross-lane reduction, ``dA`` computation, FP32->BF16 cast, and compact-NZ
  packing for ``BK=128``) executes in 256B ``T.SimdVF()`` blocks.
- **Hardware DMA Gather & L2-Resident Atomic Scatter**: Sparse ``K`` gather
  (GM->UB) uses MTE2 DMA with ``NORMAL_LV`` L2 cache policy; ``dK`` scatter-add
  (UB->GM) uses MTE3 hardware ``T.set_atomic("add", "float32")`` DMA with
  L2-allocating store policy (default ``NORMAL_FV``) so hot ``dK_fp32`` rows
  accumulate in the 128 MB L2 cache instead of bypassing to HBM.
- **L0-Resident Cube Pipeline + Staged Overlap**: ``Q`` (``a1_l0``) and ``Q^T``
  (``b3_l0``) stay resident in L0A/L0B across all K-tiles of a query; ``dQ_l0c``
  accumulates directly in L0C and drains via FixPipe FP32->BF16 straight to GM.
  Optional ``staged_pipeline=True`` overlaps Gather+GEMM1 (Stage 0),
  SIMD+GEMM2+GEMM3 (Stage 1), and MTE3 Atomic Scatter (Stage 2) across K-tiles.
- **L2 Locality & Early Exit**: Persistent contiguous query partitioning assigns
  adjacent ``(b, t)`` tokens to the same AI Core to maximize L2 reuse across
  overlapping Top-512 indices. When ``sorted_invalid_tail=True``, active tile
  count ``n_tiles`` is pre-scanned per query so trailing ``-1`` tiles and
  all-``-1`` queries skip unnecessary Cube/Vector/DMA work cleanly.
"""

from functools import lru_cache
from typing import Optional, Tuple

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

SUPPORTED_NUM_CORES = (28, 32, 36)


def build_lightning_indexer_sparse_bwd_kernel(
    H: int = 32,
    D: int = 128,
    K: int = 512,
    *,
    BK: int = 64,
    num_cores: int = 32,
    sorted_invalid_tail: bool = True,
    num_stages: int = 2,
    staged_pipeline: bool = True,
    interleaved_pipeline: bool = True,
    dk_l2_policy: str = "NORMAL_FV",
    k_l2_policy: str = "NORMAL_FV",
):
    """Build the TileLang Ascend PrimFunc for ``lightning_indexer_sparse_bwd``.

    Parameters
    ----------
    H : int
        Number of indexer heads (v1 static ``32``).
    D : int
        Indexer head dimension (v1 static ``128``).
    K : int
        Static Top-K budget (v1 static ``512``; must be divisible by ``BK``).
    BK : int
        K-tile block size, ``64`` (8 tiles) or ``128`` (4 tiles + fused compact-NZ).
    num_cores : int
        Persistent AI Core grid size (``28``, ``32`` [acceptance default], or ``36``).
    sorted_invalid_tail : bool
        If ``True``, pre-scan tile heads ``indices[b, t, tk * BK] < 0`` to
        compute the active tile count ``n_tiles`` and skip trailing ``-1`` tiles
        (effective in flat counter-mode when ``staged_pipeline=False``).
    num_stages : int
        Software pipeline stage budget for the inner K-tile loop.
    staged_pipeline : bool
        If ``True``, overlap MTE2 Gather / Cube GEMMs + SIMD / MTE3 Atomic Scatter
        across K-tiles.
    interleaved_pipeline : bool
        When ``staged_pipeline=True``, use the V2 cross-query flattened pipeline
        with explicit ping-pong buffers (zero ``% 3`` modulo, zero per-query drain
        bubbles), interleaved MTE2 gather + MTE3 atomic scatter inside the same
        8-row chunk loop (true MTE2 || MTE3 overlap), contiguous 8/4-row DMA
        coalescing, and one-time ``K_ub`` zero initialization.
    dk_l2_policy : str
        Ascend L2 cache control policy for ``dK_fp32`` MTE3 atomic stores
        (default ``"NORMAL_FV"`` to keep ``dK_fp32`` resident in L2 cache).
    k_l2_policy : str
        Ascend L2 cache control policy for ``K`` MTE2 gathers
        (default ``"NORMAL_FV"`` to keep ``K`` resident in L2 cache).
    """
    if H != 32:
        raise ValueError(f"v1 kernel locks H=32 (DeepSeek-V4.1-Flash), got H={H}")
    if D != 128:
        raise ValueError(f"v1 kernel locks D=128 (DeepSeek-V4.1-Flash), got D={D}")
    if BK not in (64, 128):
        raise ValueError(f"BK must be 64 or 128, got BK={BK}")
    if K <= 0 or K % BK != 0:
        raise ValueError(f"K ({K}) must be a positive multiple of BK ({BK})")
    if num_cores <= 0:
        raise ValueError(f"num_cores must be positive, got {num_cores}")

    NUM_K_TILES = K // BK
    ROWS = H // 2  # 16 rows per AIV (2 AIVs per AIC)
    HALF_BK = BK // 2  # 32 rows/AIV for BK=64, 64 rows/AIV for BK=128
    dtype = "bfloat16"
    accum_dtype = "float32"

    # Under 3-stage overlap (Stage 0 -> Stage 1 -> Stage 2), idx_ub is produced in
    # Stage 0 and last consumed in Stage 2 (2 iterations later), requiring 3 slots.
    idx_versions = max(num_stages, 3) if staged_pipeline else num_stages
    loop_stages = max(num_stages, 3) if staged_pipeline else num_stages
    pipe_ann = {"enable_offset": True} if staged_pipeline else None

    N_q = T.dynamic("N_q")
    N_k = T.dynamic("N_k")

    @T.macro
    def zero_k_ub(K_ub):
        """Zero K_ub [HALF_BK, 128] bf16 via 256B SIMD vector stores."""
        with T.SimdVF():
            full16 = S.pset(16, "PAT_ALL")
            zero_bf16 = T.reinterpret(S.vdup(T.uint16(0), "uint16", full16), "bfloat16x128")
            for r in T.unroll(HALF_BK):
                S.vsts(K_ub[r, 0], zero_bf16, full16, "NORM_B16")

    @T.macro
    def init_query_state(w_ub, w_scaled_ub, dw_ub, scale):
        """Pre-scale W[sid*16 : sid*16+16] by scale and zero dW accumulator."""
        with T.SimdVF():
            m16 = S.pset(32, "PAT_VL16")
            full32 = S.pset(32, "PAT_ALL")
            w_vec = S.vld(w_ub[0], "NORM")
            S.vsts(w_scaled_ub[0], S.vmuls(w_vec, scale, m16), m16, "NORM_B32")
            S.vsts(dw_ub[0], S.vdup(T.float32(0.0), "float32", full32), full32, "NORM_B32")

    @T.macro
    def vector_relu_dw_da_bk64(A_ub, idx_ub, dI_ub, w_scaled_ub, dw_ub, dA_ub, S_k_val, scale, idx_out_ub=None):
        """Fused 256B SIMD stage for BK=64: scale, ReLU, dW accumulate, dA BF16 pack."""
        with T.SimdVF():
            full32 = S.pset(32, "PAT_ALL")
            one32 = S.pset(32, "PAT_VL1")
            full16 = S.pset(16, "PAT_ALL")
            zero_f = S.vdup(T.float32(0.0), "float32", full32)

            idx_v = S.vld(idx_ub[0], "NORM")
            if idx_out_ub is not None:
                S.vsts(idx_out_ub[0], idx_v, full32, "NORM_B32")
            valid_m = S.pand(
                S.vcmps(idx_v, T.int32(0), full32, op="ge"),
                S.vcmps(idx_v, S_k_val, full32, op="lt"),
                full32,
            )
            di_v = S.vsel(S.vld(dI_ub[0], "NORM"), zero_f, valid_m)

            # Process two rows per iteration to emit one full 256B (128 bf16) store.
            for rr in T.unroll(ROWS // 2):
                r0 = rr * 2
                r1 = rr * 2 + 1

                a0 = S.vmuls(S.vld(A_ub[r0, 0], "NORM"), scale, full32)
                vp0 = S.pand(S.vcmps(a0, T.float32(0.0), full32, op="gt"), valid_m, full32)
                dw_inc0 = S.vcadd(S.vmul(a0, di_v, vp0, mode="MODE_ZEROING"), full32)
                dw_old0 = S.vld(dw_ub[r0], "BRC_B32")
                S.vsts(dw_ub[r0], S.vadd(dw_old0, dw_inc0, one32), one32, "ONEPT_B32")
                ws0 = S.vld(w_scaled_ub[r0], "BRC_B32")
                da0 = S.vmul(ws0, di_v, vp0, mode="MODE_ZEROING")

                a1 = S.vmuls(S.vld(A_ub[r1, 0], "NORM"), scale, full32)
                vp1 = S.pand(S.vcmps(a1, T.float32(0.0), full32, op="gt"), valid_m, full32)
                dw_inc1 = S.vcadd(S.vmul(a1, di_v, vp1, mode="MODE_ZEROING"), full32)
                dw_old1 = S.vld(dw_ub[r1], "BRC_B32")
                S.vsts(dw_ub[r1], S.vadd(dw_old1, dw_inc1, one32), one32, "ONEPT_B32")
                ws1 = S.vld(w_scaled_ub[r1], "BRC_B32")
                da1 = S.vmul(ws1, di_v, vp1, mode="MODE_ZEROING")

                e01, o01 = S.vdintlv(da0, da1)
                bf_e = S.vcvt(e01, dtype, full32, sat=False, part=0)
                bf_o = S.vcvt(o01, dtype, full32, sat=False, part=1)
                bf_merged = S.vor(
                    T.reinterpret(bf_e, "uint16x128"),
                    T.reinterpret(bf_o, "uint16x128"),
                    full16,
                )
                S.vsts(dA_ub[r0, 0], T.reinterpret(bf_merged, "bfloat16x128"), full16, "NORM_B16")

    @T.macro
    def vector_relu_dw_da_bk128(A_ub, idx_ub, dI_ub, w_scaled_ub, dw_ub, dA_nz_ub, S_k_val, scale, idx_out_ub=None):
        """Fused 256B SIMD stage for BK=128: scale, ReLU, dW accumulate, compact-NZ dA pack."""
        with T.SimdVF():
            full32 = S.pset(32, "PAT_ALL")
            one32 = S.pset(32, "PAT_VL1")
            b16 = S.pset(16, "PAT_VL128")
            zero_f = S.vdup(T.float32(0.0), "float32", full32)

            if idx_out_ub is not None:
                S.vsts(idx_out_ub[0], S.vld(idx_ub[0], "NORM"), full32, "NORM_B32")
                S.vsts(idx_out_ub[64], S.vld(idx_ub[64], "NORM"), full32, "NORM_B32")

            idx_e, idx_o = S.vld2(idx_ub[0], "DINTLV_B32")
            valid_e = S.pand(
                S.vcmps(idx_e, T.int32(0), full32, op="ge"),
                S.vcmps(idx_e, S_k_val, full32, op="lt"),
                full32,
            )
            valid_o = S.pand(
                S.vcmps(idx_o, T.int32(0), full32, op="ge"),
                S.vcmps(idx_o, S_k_val, full32, op="lt"),
                full32,
            )

            di_e_raw, di_o_raw = S.vld2(dI_ub[0], "DINTLV_B32")
            di_e = S.vsel(di_e_raw, zero_f, valid_e)
            di_o = S.vsel(di_o_raw, zero_f, valid_o)

            da_ptr = S.make_ubuf_ptr(
                T.access_ptr(dA_nz_ub[0, 0], "w", ROWS + 1, BK),
                dtype,
            )

            for r in T.unroll(ROWS):
                a_e_raw, a_o_raw = S.vld2(A_ub[r, 0], "DINTLV_B32")
                a_e = S.vmuls(a_e_raw, scale, full32)
                a_o = S.vmuls(a_o_raw, scale, full32)

                vp_e = S.pand(S.vcmps(a_e, T.float32(0.0), full32, op="gt"), valid_e, full32)
                vp_o = S.pand(S.vcmps(a_o, T.float32(0.0), full32, op="gt"), valid_o, full32)

                sum_e = S.vcadd(S.vmul(a_e, di_e, vp_e, mode="MODE_ZEROING"), full32)
                sum_o = S.vcadd(S.vmul(a_o, di_o, vp_o, mode="MODE_ZEROING"), full32)
                dw_inc = S.vadd(sum_e, sum_o, one32)
                dw_old = S.vld(dw_ub[r], "BRC_B32")
                S.vsts(dw_ub[r], S.vadd(dw_old, dw_inc, one32), one32, "ONEPT_B32")

                ws = S.vld(w_scaled_ub[r], "BRC_B32")
                da_e = S.vmul(ws, di_e, vp_e, mode="MODE_ZEROING")
                da_o = S.vmul(ws, di_o, vp_o, mode="MODE_ZEROING")

                da_e_bf16 = S.vcvt(da_e, dtype, full32, sat=False, part=0)
                da_o_bf16 = S.vcvt(da_o, dtype, full32, sat=False, part=1)
                da_merged = S.vor(
                    T.reinterpret(da_e_bf16, "uint16x128"),
                    T.reinterpret(da_o_bf16, "uint16x128"),
                    b16,
                )
                da_ptr = S.vsstb(
                    T.reinterpret(da_merged, "bfloat16x128"),
                    da_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    b16,
                    update=True,
                )

    @T.macro
    def stage_gather_and_gemm1(
        q_idx, k_off, k_base, sid, Indices, dI, K_gm, S_k_val,
        idx_ub, dI_ub, K_ub, K_l1, a1_l0, b1_l0, A_l0c, A_ub,
    ):
        T.copy(Indices[q_idx, k_off : k_off + BK], idx_ub[0:BK])
        T.copy(dI[q_idx, k_off : k_off + BK], dI_ub[0:BK])
        zero_k_ub(K_ub)
        for r in T.serial(HALF_BK):
            idx_r = idx_ub[sid * HALF_BK + r]
            if idx_r >= 0 and idx_r < S_k_val:
                T.copy(K_gm[k_base + idx_r, 0:D], K_ub[r, 0:D], l2_cache_ctrl=k_l2_policy)
        T.dual_copy(K_ub, K_l1)

        for _ in range(1):
            T.copy(K_l1, b1_l0)
            T.gemm(a1_l0, b1_l0, A_l0c, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)
            T.dual_copy(A_l0c, A_ub, unit_flag_ctrl=3)

    @T.macro
    def analyze_idx_contiguity(idx_ub, meta_ub, sid, S_k_val):
        """256B SIMD contiguity + validity analyzer for HALF_BK rows (zero PIPE_S scan).

        Writes 8 float32 summary flags into ``meta_ub[0..7]`` (reused by Stage 1 & Stage 4):
          - ``meta_ub[0]`` == 0.0 iff ALL ``HALF_BK`` rows are valid in ``[0, S_k_val)`` AND contiguous.
          - ``meta_ub[1..4]`` == 0.0 iff slice ``s`` (``SLICE_ROWS`` rows) is valid AND contiguous.
          - ``meta_ub[5]`` == count of contiguous even-odd adjacent pairs (``< 2.0`` => pure scatter).
          - ``meta_ub[6]`` == count of invalid (``< 0`` or ``>= S_k_val``) rows
              (``0.0`` => branchless 4-way scatter; ``float(HALF_BK)`` => O(1) skip).
          - ``meta_ub[7]`` == 0.0 iff ``[0..4)`` AND ``[4..HALF_BK)`` are each contiguous & valid
              (2-burst fast path for ``sink_tokens=4`` + contiguous window).
        """
        with T.SimdVF():
            full32 = S.pset(32, "PAT_ALL")
            one32 = S.pset(32, "PAT_VL1")
            zero_f = S.vdup(T.float32(0.0), "float32", full32)
            one_f = S.vdup(T.float32(1.0), "float32", full32)

            v_idx = S.vld(idx_ub[sid * HALF_BK], "NORM")
            v_iota = S.vci(T.int32(0), "int32", "INC_ORDER")
            v_diff = S.vsub(v_idx, v_iota, full32)

            if BK == 128:
                active_m = full32
                pair_m = S.pset(32, "PAT_VL32")
                m0 = S.vcmps(v_iota, T.int32(16), full32, op="lt")
                m_lt32 = S.vcmps(v_iota, T.int32(32), full32, op="lt")
                m1 = S.pxor(m_lt32, m0, full32)
                m_lt48 = S.vcmps(v_iota, T.int32(48), full32, op="lt")
                m2 = S.pxor(m_lt48, m_lt32, full32)
                m3 = S.pnot(m_lt48, full32)
            else:
                active_m = S.pset(32, "PAT_VL32")
                pair_m = S.pset(32, "PAT_VL16")
                m0 = S.vcmps(v_iota, T.int32(8), active_m, op="lt")
                m_lt16 = S.vcmps(v_iota, T.int32(16), active_m, op="lt")
                m1 = S.pxor(m_lt16, m0, active_m)
                m_lt24 = S.vcmps(v_iota, T.int32(24), active_m, op="lt")
                m2 = S.pxor(m_lt24, m_lt16, active_m)
                m3 = S.pxor(active_m, m_lt24, active_m)

            valid_m = S.pand(
                S.vcmps(v_idx, T.int32(0), active_m, op="ge"),
                S.vcmps(v_idx, S_k_val, active_m, op="lt"),
                active_m,
            )
            bad_valid = S.vcadd(S.vsel(zero_f, one_f, valid_m), active_m)
            S.vsts(meta_ub[6], bad_valid, one32, "ONEPT_B32")

            v_b0 = S.vdupv(v_diff, active_m, "POS_LOWEST")
            eq_all = S.vcmp(v_diff, v_b0, active_m, op="eq")
            bad_all = S.vcadd(S.vsel(zero_f, one_f, eq_all), active_m)
            S.vsts(meta_ub[0], S.vadd(bad_all, bad_valid, one32), one32, "ONEPT_B32")

            # Sink4 + contiguous window fast path: [0..4) contiguous & [4..HALF_BK) contiguous, all valid.
            m_head4 = S.vcmps(v_iota, T.int32(4), active_m, op="lt")
            m_tail4 = S.pxor(active_m, m_head4, active_m)
            bad_h4 = S.vcadd(S.vsel(zero_f, one_f, S.pand(eq_all, valid_m, m_head4)), m_head4)
            v_b4 = S.vadds(S.vld(idx_ub[sid * HALF_BK + 4], "BRC_B32"), T.int32(-4), active_m)
            ok_t4 = S.pand(S.vcmp(v_diff, v_b4, m_tail4, op="eq"), valid_m, m_tail4)
            bad_t4 = S.vcadd(S.vsel(zero_f, one_f, ok_t4), m_tail4)
            S.vsts(meta_ub[7], S.vadd(bad_h4, bad_t4, one32), one32, "ONEPT_B32")

            ok_s0 = S.pand(eq_all, valid_m, m0)
            S.vsts(meta_ub[1], S.vcadd(S.vsel(zero_f, one_f, ok_s0), m0), one32, "ONEPT_B32")

            SLICE_STEP = HALF_BK // 4
            v_b1 = S.vadds(S.vld(idx_ub[sid * HALF_BK + SLICE_STEP], "BRC_B32"), T.int32(-SLICE_STEP), active_m)
            ok_s1 = S.pand(S.vcmp(v_diff, v_b1, m1, op="eq"), valid_m, m1)
            S.vsts(meta_ub[2], S.vcadd(S.vsel(zero_f, one_f, ok_s1), m1), one32, "ONEPT_B32")

            v_b2 = S.vadds(S.vld(idx_ub[sid * HALF_BK + 2 * SLICE_STEP], "BRC_B32"), T.int32(-2 * SLICE_STEP), active_m)
            ok_s2 = S.pand(S.vcmp(v_diff, v_b2, m2, op="eq"), valid_m, m2)
            S.vsts(meta_ub[3], S.vcadd(S.vsel(zero_f, one_f, ok_s2), m2), one32, "ONEPT_B32")

            v_b3 = S.vadds(S.vld(idx_ub[sid * HALF_BK + 3 * SLICE_STEP], "BRC_B32"), T.int32(-3 * SLICE_STEP), active_m)
            ok_s3 = S.pand(S.vcmp(v_diff, v_b3, m3, op="eq"), valid_m, m3)
            S.vsts(meta_ub[4], S.vcadd(S.vsel(zero_f, one_f, ok_s3), m3), one32, "ONEPT_B32")

            e_diff, o_diff = S.vdintlv(v_diff, v_diff)
            pair_eq = S.vcmp(e_diff, o_diff, pair_m, op="eq")
            n_pairs = S.vcadd(S.vsel(one_f, zero_f, pair_eq), pair_m)
            S.vsts(meta_ub[5], n_pairs, one32, "ONEPT_B32")

    @T.macro
    def gather_tile_v3(K_gm, k_base, idx_ub, meta_ub, K_ub, sid, S_k_val):
        """SIMD-guided MTE2 gather: 64/32-row full burst, 4+(HALF_BK-4) sink+window 2-burst,
        4-way prefetched pure scatter, or 16/8/4-row coalesced."""
        SLICE_STEP = HALF_BK // 4
        if meta_ub[0] == T.float32(0.0):
            i0 = idx_ub[sid * HALF_BK]
            T.copy(
                K_gm[k_base + i0 : k_base + i0 + HALF_BK, 0:D],
                K_ub[0:HALF_BK, 0:D],
                l2_cache_ctrl=k_l2_policy,
            )
        else:
            if meta_ub[7] == T.float32(0.0):
                i0 = idx_ub[sid * HALF_BK]
                i4 = idx_ub[sid * HALF_BK + 4]
                T.copy(
                    K_gm[k_base + i0 : k_base + i0 + 4, 0:D],
                    K_ub[0:4, 0:D],
                    l2_cache_ctrl=k_l2_policy,
                )
                T.copy(
                    K_gm[k_base + i4 : k_base + i4 + (HALF_BK - 4), 0:D],
                    K_ub[4:HALF_BK, 0:D],
                    l2_cache_ctrl=k_l2_policy,
                )
            else:
                if meta_ub[5] < T.float32(2.0):
                    if meta_ub[6] == T.float32(0.0):
                        for r4 in T.serial(HALF_BK // 4):
                            rb = r4 * 4
                            ib = sid * HALF_BK + rb
                            idx0 = idx_ub[ib + 0]
                            idx1 = idx_ub[ib + 1]
                            idx2 = idx_ub[ib + 2]
                            idx3 = idx_ub[ib + 3]
                            T.copy(K_gm[k_base + idx0, 0:D], K_ub[rb + 0, 0:D], l2_cache_ctrl=k_l2_policy)
                            T.copy(K_gm[k_base + idx1, 0:D], K_ub[rb + 1, 0:D], l2_cache_ctrl=k_l2_policy)
                            T.copy(K_gm[k_base + idx2, 0:D], K_ub[rb + 2, 0:D], l2_cache_ctrl=k_l2_policy)
                            T.copy(K_gm[k_base + idx3, 0:D], K_ub[rb + 3, 0:D], l2_cache_ctrl=k_l2_policy)
                    else:
                        if meta_ub[6] < T.float32(HALF_BK):
                            for r in T.serial(HALF_BK):
                                idx_r = idx_ub[sid * HALF_BK + r]
                                if idx_r >= 0 and idx_r < S_k_val:
                                    T.copy(K_gm[k_base + idx_r, 0:D], K_ub[r, 0:D], l2_cache_ctrl=k_l2_policy)
                else:
                    for s in T.serial(4):
                        r_start = s * SLICE_STEP
                        base_idx = sid * HALF_BK + r_start
                        if meta_ub[1 + s] == T.float32(0.0):
                            is0 = idx_ub[base_idx]
                            T.copy(
                                K_gm[k_base + is0 : k_base + is0 + SLICE_STEP, 0:D],
                                K_ub[r_start : r_start + SLICE_STEP, 0:D],
                                l2_cache_ctrl=k_l2_policy,
                            )
                        else:
                            for sub in T.serial(SLICE_STEP // 4):
                                r_sub = r_start + sub * 4
                                b_sub = base_idx + sub * 4
                                a0 = idx_ub[b_sub + 0]
                                a1 = idx_ub[b_sub + 1]
                                a2 = idx_ub[b_sub + 2]
                                a3 = idx_ub[b_sub + 3]
                                if a0 >= 0 and a0 + 4 <= S_k_val and a1 == a0 + 1 and a2 == a0 + 2 and a3 == a0 + 3:
                                    T.copy(
                                        K_gm[k_base + a0 : k_base + a0 + 4, 0:D],
                                        K_ub[r_sub : r_sub + 4, 0:D],
                                        l2_cache_ctrl=k_l2_policy,
                                    )
                                else:
                                    for j in T.serial(4):
                                        idx_j = idx_ub[b_sub + j]
                                        if idx_j >= 0 and idx_j < S_k_val:
                                            T.copy(
                                                K_gm[k_base + idx_j, 0:D],
                                                K_ub[r_sub + j, 0:D],
                                                l2_cache_ctrl=k_l2_policy,
                                            )

    @T.macro
    def scatter_tile_v3(dK_ub, dK_fp32, k_base, idx_ub, meta_ub, sid, S_k_val):
        """SIMD-guided MTE3 FP32 atomic scatter: 64/32-row full burst, 4+(HALF_BK-4) sink+window 2-burst,
        4-way prefetched pure scatter, or 16/8/4-row coalesced."""
        SLICE_STEP = HALF_BK // 4
        if meta_ub[0] == T.float32(0.0):
            j0 = idx_ub[sid * HALF_BK]
            T.assume_no_conflict(dK_fp32)
            T.copy(
                dK_ub[0:HALF_BK, 0:D],
                dK_fp32[k_base + j0 : k_base + j0 + HALF_BK, 0:D],
                l2_cache_ctrl=dk_l2_policy,
            )
        else:
            if meta_ub[7] == T.float32(0.0):
                j0 = idx_ub[sid * HALF_BK]
                j4 = idx_ub[sid * HALF_BK + 4]
                T.assume_no_conflict(dK_fp32)
                T.copy(
                    dK_ub[0:4, 0:D],
                    dK_fp32[k_base + j0 : k_base + j0 + 4, 0:D],
                    l2_cache_ctrl=dk_l2_policy,
                )
                T.assume_no_conflict(dK_fp32)
                T.copy(
                    dK_ub[4:HALF_BK, 0:D],
                    dK_fp32[k_base + j4 : k_base + j4 + (HALF_BK - 4), 0:D],
                    l2_cache_ctrl=dk_l2_policy,
                )
            else:
                if meta_ub[5] < T.float32(2.0):
                    if meta_ub[6] == T.float32(0.0):
                        for r4 in T.serial(HALF_BK // 4):
                            rb = r4 * 4
                            ib = sid * HALF_BK + rb
                            idx0 = idx_ub[ib + 0]
                            idx1 = idx_ub[ib + 1]
                            idx2 = idx_ub[ib + 2]
                            idx3 = idx_ub[ib + 3]
                            T.assume_no_conflict(dK_fp32)
                            T.copy(dK_ub[rb + 0, 0:D], dK_fp32[k_base + idx0, 0:D], l2_cache_ctrl=dk_l2_policy)
                            T.assume_no_conflict(dK_fp32)
                            T.copy(dK_ub[rb + 1, 0:D], dK_fp32[k_base + idx1, 0:D], l2_cache_ctrl=dk_l2_policy)
                            T.assume_no_conflict(dK_fp32)
                            T.copy(dK_ub[rb + 2, 0:D], dK_fp32[k_base + idx2, 0:D], l2_cache_ctrl=dk_l2_policy)
                            T.assume_no_conflict(dK_fp32)
                            T.copy(dK_ub[rb + 3, 0:D], dK_fp32[k_base + idx3, 0:D], l2_cache_ctrl=dk_l2_policy)
                    else:
                        if meta_ub[6] < T.float32(HALF_BK):
                            for r in T.serial(HALF_BK):
                                idx_r = idx_ub[sid * HALF_BK + r]
                                if idx_r >= 0 and idx_r < S_k_val:
                                    T.assume_no_conflict(dK_fp32)
                                    T.copy(dK_ub[r, 0:D], dK_fp32[k_base + idx_r, 0:D], l2_cache_ctrl=dk_l2_policy)
                else:
                    for s in T.serial(4):
                        r_start = s * SLICE_STEP
                        base_idx = sid * HALF_BK + r_start
                        if meta_ub[1 + s] == T.float32(0.0):
                            js0 = idx_ub[base_idx]
                            T.assume_no_conflict(dK_fp32)
                            T.copy(
                                dK_ub[r_start : r_start + SLICE_STEP, 0:D],
                                dK_fp32[k_base + js0 : k_base + js0 + SLICE_STEP, 0:D],
                                l2_cache_ctrl=dk_l2_policy,
                            )
                        else:
                            for sub in T.serial(SLICE_STEP // 4):
                                r_sub = r_start + sub * 4
                                b_sub = base_idx + sub * 4
                                a0 = idx_ub[b_sub + 0]
                                a1 = idx_ub[b_sub + 1]
                                a2 = idx_ub[b_sub + 2]
                                a3 = idx_ub[b_sub + 3]
                                if a0 >= 0 and a0 + 4 <= S_k_val and a1 == a0 + 1 and a2 == a0 + 2 and a3 == a0 + 3:
                                    T.assume_no_conflict(dK_fp32)
                                    T.copy(
                                        dK_ub[r_sub : r_sub + 4, 0:D],
                                        dK_fp32[k_base + a0 : k_base + a0 + 4, 0:D],
                                        l2_cache_ctrl=dk_l2_policy,
                                    )
                                else:
                                    for j in T.serial(4):
                                        idx_j = idx_ub[b_sub + j]
                                        if idx_j >= 0 and idx_j < S_k_val:
                                            T.assume_no_conflict(dK_fp32)
                                            T.copy(
                                                dK_ub[r_sub + j, 0:D],
                                                dK_fp32[k_base + idx_j, 0:D],
                                                l2_cache_ctrl=dk_l2_policy,
                                            )

    @T.macro
    def run_gemm1_only(K_l1, a1_l0, b1_l0, A_l0c, A_ub):
        for _ in range(1):
            T.copy(K_l1, b1_l0)
            T.gemm(a1_l0, b1_l0, A_l0c, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)
            T.dual_copy(A_l0c, A_ub, unit_flag_ctrl=3)

    @T.macro
    def run_stage1_vec_gemm23(
        clear_dq, S_k_val, scale,
        A_ub, idx_ub, dI_ub, w_scaled_ub, dw_ub, dA_store_ub, dA_l1, K_l1,
        a2_l0, b2_l0, dQ_l0c, a3_l0, b3_l0, dK_l0c, dK_ub, idx_out_ub,
    ):
        if BK == 128:
            vector_relu_dw_da_bk128(A_ub, idx_ub, dI_ub, w_scaled_ub, dw_ub, dA_store_ub, S_k_val, scale, idx_out_ub)
            T.dual_copy(dA_store_ub[0:ROWS, 0:BK], dA_l1)
        else:
            vector_relu_dw_da_bk64(A_ub, idx_ub, dI_ub, w_scaled_ub, dw_ub, dA_store_ub, S_k_val, scale, idx_out_ub)
            T.dual_copy(dA_store_ub, dA_l1)

        for _ in range(1):
            T.copy(dA_l1, a3_l0, transpose=True)
            T.gemm(a3_l0, b3_l0, dK_l0c, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)
            T.dual_copy(dK_l0c, dK_ub, unit_flag_ctrl=3)

        for _ in range(1):
            T.copy(dA_l1, a2_l0)
            T.copy(K_l1, b2_l0, transpose=True)
            T.gemm(
                a2_l0,
                b2_l0,
                dQ_l0c,
                transpose_B=True,
                clear_accum=clear_dq,
            )

    @T.macro
    def stage_vec_and_gemm23(
        tile_k, S_k_val, scale,
        A_ub, idx_ub, dI_ub, w_scaled_ub, dw_ub, dA_store_ub, dA_l1, K_l1,
        a2_l0, b2_l0, dQ_l0c, a3_l0, b3_l0, dK_l0c, dK_ub,
    ):
        if BK == 128:
            vector_relu_dw_da_bk128(A_ub, idx_ub, dI_ub, w_scaled_ub, dw_ub, dA_store_ub, S_k_val, scale)
            T.dual_copy(dA_store_ub[0:ROWS, 0:BK], dA_l1)
        else:
            vector_relu_dw_da_bk64(A_ub, idx_ub, dI_ub, w_scaled_ub, dw_ub, dA_store_ub, S_k_val, scale)
            T.dual_copy(dA_store_ub, dA_l1)

        # GEMM 3 first so FixPipe -> dK_ub starts immediately while GEMM 2 accumulates dQ_l0c.
        for _ in range(1):
            T.copy(dA_l1, a3_l0, transpose=True)
            T.gemm(a3_l0, b3_l0, dK_l0c, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)
            T.dual_copy(dK_l0c, dK_ub, unit_flag_ctrl=3)

        for _ in range(1):
            T.copy(dA_l1, a2_l0)
            T.copy(K_l1, b2_l0, transpose=True)
            T.gemm(
                a2_l0,
                b2_l0,
                dQ_l0c,
                transpose_B=True,
                clear_accum=(tile_k == 0),
            )

    @T.macro
    def stage_scatter_dk(k_base, sid, S_k_val, idx_ub, dK_ub, dK_fp32):
        # Hardware FP32 atomic-add is commutative and per-query indices are unique,
        # so AIV0/AIV1 and cross-iteration atomic stores have no ordering hazard.
        for r in T.serial(HALF_BK):
            idx_r = idx_ub[sid * HALF_BK + r]
            if idx_r >= 0 and idx_r < S_k_val:
                T.assume_no_conflict(dK_fp32)
                T.copy(dK_ub[r, 0:D], dK_fp32[k_base + idx_r, 0:D], l2_cache_ctrl=dk_l2_policy)

    if staged_pipeline and interleaved_pipeline:
        # V4 Single flattened 1D 5-stage T.Pipelined loop across ALL (q_count * NUM_K_TILES) tiles:
        #  - Stage 0 (Prefetch + SIMD Analyze): streaming NOTALLOC_DROP loads for Q/W/Indices/dI +
        #    256B SIMD contiguity analyzer (meta_ub) — ZERO PIPE_S wait, Q prefetched 2 stages ahead of GEMM1!
        #  - Stage 1 (Pure AIV MTE2 Gather): meta_ub already ready from Stage 0 (zero PIPE_V->PIPE_S stall!),
        #    64/32-row full burst, 4+(HALF_BK-4) sink+window 2-burst, or 4-way prefetched scatter (ZERO MTE3!).
        #  - Stage 2 (MTE3 + AIC GEMM1): Q_l1->a1_l0 (zero HBM wait!), K_ub->K_l1 (already in UB!), GEMM1.
        #  - Stage 3 (V + MTE3 + AIC GEMM3/2 + Direct AIC dQ): Q_l1->b3_l0, w_scaled, 256B SIMD, GEMM3/2,
        #    direct AIC FixPipe FP32->BF16 dQ store (NOTALLOC_CI, eliminates AIV dQ cast + PIPE_MTE3 barrier!).
        #  - Stage 4 (AIV MTE3 Atomic Scatter + dW): dK_ub -> dK_fp32 (L2-resident NORMAL_FV) + dW store.
        TK_SHIFT = 2 if BK == 128 else 3
        TK_MASK = NUM_K_TILES - 1

        @T.prim_func
        def main_v2(
            Q: T.Tensor((N_q, H, D), dtype),
            K_gm: T.Tensor((N_k, D), dtype),
            W: T.Tensor((N_q, H), accum_dtype),
            Indices: T.Tensor((N_q, K), "int32"),
            dI: T.Tensor((N_q, K), accum_dtype),
            dQ: T.Tensor((N_q, H, D), dtype),
            dK_fp32: T.Tensor((N_k, D), accum_dtype),
            dW: T.Tensor((N_q, H), accum_dtype),
            S_q_val: T.int32,
            S_k_val: T.int32,
            scale: T.float32,
        ):
            with T.MixedKernel(num_cores, sids=2) as (bx, sid):
                # ----- L1 buffers -----
                Q_l1 = T.alloc_l1((H, D), dtype)
                K_l1 = T.alloc_l1((BK, D), dtype)
                dA_l1 = T.alloc_l1((H, BK), dtype)

                # ----- L0A / L0B / L0C buffers -----
                a1_l0 = T.alloc_l0a((H, D), dtype)
                a2_l0 = T.alloc_l0a((H, BK), dtype)
                a3_l0 = T.alloc_l0a((BK, H), dtype)

                b1_l0 = T.alloc_l0b((BK, D), dtype)
                b2_l0 = T.view(b1_l0, shape=(D, BK))
                b3_l0 = T.alloc_l0b((D, H), dtype)

                A_l0c = T.alloc_l0c((H, BK), accum_dtype)
                dQ_l0c = T.alloc_l0c((H, D), accum_dtype)
                dK_l0c = T.alloc_l0c((BK, D), accum_dtype)

                # ----- UB buffers per AIV -----
                K_ub = T.alloc_shared((HALF_BK, D), dtype)
                A_ub = T.alloc_shared((ROWS, BK), accum_dtype)
                dK_ub = T.alloc_shared((HALF_BK, D), accum_dtype)
                idx_ub = T.alloc_shared((128,), "int32")  # padded to 128 so sid*32+64 stays in-bounds for BK=64
                meta_ub = T.alloc_shared((64,), accum_dtype)
                dI_ub = T.alloc_shared((BK,), accum_dtype)
                w_ub = T.alloc_shared((64,), accum_dtype)
                w_scaled_ub = T.alloc_shared((64,), accum_dtype)
                dw_ub = T.alloc_shared((64,), accum_dtype)
                dw_out_ub = T.alloc_shared((64,), accum_dtype)

                if BK == 128:
                    dA_store_ub = T.alloc_shared((ROWS + 1, BK), dtype)
                    T.annotate_layout({dA_store_ub: make_ascend_compact_nz_layout(dA_store_ub)})
                else:
                    dA_store_ub = T.alloc_shared((ROWS, BK), dtype)

                T.annotate_buffer_versions(
                    {
                        Q_l1: 4,      # Stage 0 -> Stage 3 (& 3, 32 KiB L1)
                        a1_l0: 1,
                        b3_l0: 1,
                        dQ_l0c: 1,
                        w_ub: 2,      # Stage 2 -> Stage 3 (& 1; keeps V_MTE2 flags <= 8, zero lock spill!)
                        w_scaled_ub: 1,
                        dw_ub: 1,
                        dw_out_ub: 2,
                        b1_l0: 2 if BK == 64 else (1, "auto"),
                        K_l1: 2,
                        dA_l1: 2,
                        a2_l0: 2,
                        a3_l0: 2,
                        A_l0c: 2,
                        dK_l0c: 2,
                        K_ub: 2,
                        A_ub: 2,
                        dK_ub: 2,
                        idx_ub: 8,    # power-of-2 (& 7), covers Stage 0 -> Stage 4 (4 KiB UB)
                        meta_ub: 8,   # power-of-2 (& 7), covers Stage 0 -> Stage 4 (2 KiB UB)
                        dI_ub: 4,     # power-of-2 (& 3), covers Stage 0 -> Stage 3 (2 KiB UB)
                        dA_store_ub: 2,
                    }
                )

                # Hoisted scalar partition bounds (eliminates inner-loop N_q / num_cores).
                q_per_core = T.alloc_var("int32", init=N_q // num_cores)
                q_rem = T.alloc_var("int32", init=N_q % num_cores)
                q_start = T.alloc_var("int32", init=bx * q_per_core + T.min(bx, q_rem))
                q_count = T.alloc_var("int32", init=q_per_core + T.if_then_else(bx < q_rem, 1, 0))
                total_tiles = T.alloc_var("int32", init=q_count * NUM_K_TILES)
                is_single_batch = T.alloc_var("int32", init=T.if_then_else(N_q == S_q_val, 1, 0))

                for step in T.Pipelined(total_tiles, num_stages=5, annotations={"enable_offset": True}):
                    qi = step >> TK_SHIFT
                    tk = step & TK_MASK
                    q_idx = q_start + qi
                    k_off = tk * BK
                    k_base = T.if_then_else(is_single_batch == 1, 0, (q_idx // S_q_val) * S_k_val)

                    # Stage 0: Streaming no-L2-allocate prefetch of Q/Indices/dI + SIMD contiguity analysis.
                    # Decoupling Stage 0 from Stage 1 eliminates BOTH the Q cold-HBM stall AND the PIPE_V->PIPE_S stall!
                    with T.Stage(0):
                        if tk == 0:
                            T.copy(Q[q_idx, 0:H, 0:D], Q_l1, l2_cache_ctrl="NOTALLOC_DROP")
                        T.copy(Indices[q_idx, k_off : k_off + BK], idx_ub[0:BK], l2_cache_ctrl="NOTALLOC_DROP")
                        analyze_idx_contiguity(idx_ub, meta_ub, sid, S_k_val)
                        T.copy(dI[q_idx, k_off : k_off + BK], dI_ub[0:BK], l2_cache_ctrl="NOTALLOC_DROP")

                    # Stage 1: Pure AIV MTE2 K gather using meta_ub already ready from Stage 0 (zero MTE3!).
                    with T.Stage(1):
                        if step < 2:
                            zero_k_ub(K_ub)
                        for _once in T.serial(T.min(1, S_k_val)):
                            gather_tile_v3(K_gm, k_base, idx_ub, meta_ub, K_ub, sid, S_k_val)

                    # Stage 2: Load Q_l1 -> a1_l0 (already prefetched 2 stages ago!), prefetch W -> w_ub (1 stage ahead),
                    # ship K_ub -> K_l1 (already in UB!), and run GEMM1.
                    with T.Stage(2):
                        if tk == 0:
                            T.copy(Q_l1, a1_l0)
                            T.copy(W[q_idx, sid * ROWS : (sid + 1) * ROWS], w_ub[0:ROWS], l2_cache_ctrl="NOTALLOC_DROP")
                        T.dual_copy(K_ub, K_l1)
                        run_gemm1_only(K_l1, a1_l0, b1_l0, A_l0c, A_ub)

                    # Stage 3: Switch Q^T/w_scaled at tk==0, run 256B SIMD + GEMM3 + GEMM2,
                    # direct AIC FixPipe FP32->BF16 dQ drain (NOTALLOC_CI) + copy dw_ub -> dw_out_ub.
                    with T.Stage(3):
                        if tk == 0:
                            T.copy(Q_l1, b3_l0, transpose=True)
                            init_query_state(w_ub, w_scaled_ub, dw_ub, scale)

                        run_stage1_vec_gemm23(
                            tk == 0, S_k_val, scale,
                            A_ub, idx_ub, dI_ub, w_scaled_ub, dw_ub, dA_store_ub, dA_l1, K_l1,
                            a2_l0, b2_l0, dQ_l0c, a3_l0, b3_l0, dK_l0c, dK_ub, None,
                        )

                        if tk == NUM_K_TILES - 1:
                            T.copy(dQ_l0c, dQ[q_idx, 0:H, 0:D], l2_cache_ctrl="NOTALLOC_CI")
                            with T.SimdVF():
                                m16 = S.pset(32, "PAT_VL16")
                                S.vsts(dw_out_ub[0], S.vld(dw_ub[0], "NORM"), m16, "NORM_B32")

                    # Stage 4: SIMD-guided MTE3 FP32 atomic scatter of dK_ub -> dK_fp32 (L2-resident),
                    # followed by non-atomic dW store cleanly after set_atomic_none().
                    with T.Stage(4):
                        T.set_atomic("add", accum_dtype)
                        for _once in T.serial(T.min(1, S_k_val)):
                            scatter_tile_v3(dK_ub, dK_fp32, k_base, idx_ub, meta_ub, sid, S_k_val)
                        T.set_atomic_none()
                        if tk == NUM_K_TILES - 1:
                            T.copy(dw_out_ub[0:ROWS], dW[q_idx, sid * ROWS : (sid + 1) * ROWS], l2_cache_ctrl="NOTALLOC_CI")

        return main_v2

    @T.prim_func
    def main(
        Q: T.Tensor((N_q, H, D), dtype),
        K_gm: T.Tensor((N_k, D), dtype),
        W: T.Tensor((N_q, H), accum_dtype),
        Indices: T.Tensor((N_q, K), "int32"),
        dI: T.Tensor((N_q, K), accum_dtype),
        dQ: T.Tensor((N_q, H, D), dtype),
        dK_fp32: T.Tensor((N_k, D), accum_dtype),
        dW: T.Tensor((N_q, H), accum_dtype),
        S_q_val: T.int32,
        S_k_val: T.int32,
        scale: T.float32,
    ):
        with T.MixedKernel(num_cores, sids=2) as (bx, sid):
            # ----- L1 buffers (512 KiB budget) -----
            Q_l1 = T.alloc_l1((H, D), dtype)
            K_l1 = T.alloc_l1((BK, D), dtype)
            dA_l1 = T.alloc_l1((H, BK), dtype)

            # ----- L0A buffers (64 KiB budget) -----
            a1_l0 = T.alloc_l0a((H, D), dtype)
            a2_l0 = T.alloc_l0a((H, BK), dtype)
            a3_l0 = T.alloc_l0a((BK, H), dtype)

            # ----- L0B buffers (64 KiB budget) -----
            b1_l0 = T.alloc_l0b((BK, D), dtype)
            b2_l0 = T.view(b1_l0, shape=(D, BK))
            b3_l0 = T.alloc_l0b((D, H), dtype)

            # ----- L0C buffers (256 KiB budget) -----
            A_l0c = T.alloc_l0c((H, BK), accum_dtype)
            dQ_l0c = T.alloc_l0c((H, D), accum_dtype)
            dK_l0c = T.alloc_l0c((BK, D), accum_dtype)

            # ----- UB buffers per AIV (248 KiB budget) -----
            K_ub = T.alloc_shared((HALF_BK, D), dtype)
            zero_dq_ub = T.alloc_shared((ROWS, D), dtype)
            A_ub = T.alloc_shared((ROWS, BK), accum_dtype)
            dK_ub = T.alloc_shared((HALF_BK, D), accum_dtype)
            idx_ub = T.alloc_shared((BK,), "int32")
            dI_ub = T.alloc_shared((BK,), accum_dtype)
            w_ub = T.alloc_shared((64,), accum_dtype)
            w_scaled_ub = T.alloc_shared((64,), accum_dtype)
            dw_ub = T.alloc_shared((64,), accum_dtype)

            if BK == 128:
                dA_store_ub = T.alloc_shared((ROWS + 1, BK), dtype)
                T.annotate_layout({dA_store_ub: make_ascend_compact_nz_layout(dA_store_ub)})
            else:
                dA_store_ub = T.alloc_shared((ROWS, BK), dtype)

            T.annotate_buffer_versions(
                {
                    Q_l1: 1,
                    a1_l0: 1,
                    b3_l0: 1,
                    dQ_l0c: 1,
                    zero_dq_ub: 1,
                    w_ub: 1,
                    w_scaled_ub: 1,
                    dw_ub: 1,
                    b1_l0: 2 if BK == 64 else (1, "auto"),
                    K_l1: num_stages,
                    dA_l1: num_stages,
                    a2_l0: num_stages,
                    a3_l0: num_stages,
                    A_l0c: num_stages,
                    dK_l0c: num_stages,
                    K_ub: num_stages,
                    A_ub: num_stages,
                    dK_ub: num_stages,
                    idx_ub: idx_versions,
                    dI_ub: num_stages,
                    dA_store_ub: num_stages,
                }
            )

            # Persistent contiguous query partitioning for L2 cache locality.
            q_per_core = N_q // num_cores
            q_rem = N_q % num_cores
            q_start = bx * q_per_core + T.min(bx, q_rem)
            q_count = q_per_core + T.if_then_else(bx < q_rem, 1, 0)

            cur_t = T.alloc_var("int32", init=q_start % S_q_val)
            cur_k_base = T.alloc_var("int32", init=(q_start // S_q_val) * S_k_val)

            for qi in T.serial(q_count):
                q_idx = q_start + qi
                k_base = T.alloc_var("int32", init=cur_k_base)

                if staged_pipeline:
                    # Unconditional constant-bound T.Pipelined(NUM_K_TILES) keeps
                    # iteration-mode buffer versioning strictly affine across queries
                    # (zero non-affine warnings, 100% deadlock-free 3-stage overlap).
                    T.copy(Q[q_idx, 0:H, 0:D], Q_l1)
                    T.copy(Q_l1, a1_l0)
                    T.copy(Q_l1, b3_l0, transpose=True)

                    T.copy(W[q_idx, sid * ROWS : (sid + 1) * ROWS], w_ub[0:ROWS])
                    init_query_state(w_ub, w_scaled_ub, dw_ub, scale)

                    T.set_atomic("add", accum_dtype)

                    for tile_k in T.Pipelined(NUM_K_TILES, num_stages=loop_stages, annotations=pipe_ann):
                        k_off = tile_k * BK
                        with T.Stage(0):
                            stage_gather_and_gemm1(
                                q_idx, k_off, k_base, sid, Indices, dI, K_gm, S_k_val,
                                idx_ub, dI_ub, K_ub, K_l1, a1_l0, b1_l0, A_l0c, A_ub,
                            )
                        with T.Stage(1):
                            stage_vec_and_gemm23(
                                tile_k, S_k_val, scale,
                                A_ub, idx_ub, dI_ub, w_scaled_ub, dw_ub, dA_store_ub, dA_l1, K_l1,
                                a2_l0, b2_l0, dQ_l0c, a3_l0, b3_l0, dK_l0c, dK_ub,
                            )
                        with T.Stage(2):
                            stage_scatter_dk(k_base, sid, S_k_val, idx_ub, dK_ub, dK_fp32)

                    T.set_atomic_none()

                    T.copy(dQ_l0c, dQ[q_idx, 0:H, 0:D])
                    T.copy(dw_ub[0:ROWS], dW[q_idx, sid * ROWS : (sid + 1) * ROWS])
                else:
                    # Counter-mode pipelining supports variable n_tiles & all-(-1) early exit
                    # with zero non-affine warnings.
                    n_tiles = T.alloc_var("int32", init=0)
                    if sorted_invalid_tail:
                        for tk in T.unroll(NUM_K_TILES):
                            if n_tiles == tk:
                                if Indices[q_idx, tk * BK] >= 0:
                                    n_tiles = tk + 1
                    else:
                        n_tiles = NUM_K_TILES

                    if n_tiles == 0:
                        with T.SimdVF():
                            full16 = S.pset(16, "PAT_ALL")
                            full32 = S.pset(32, "PAT_ALL")
                            zero_bf16 = T.reinterpret(S.vdup(T.uint16(0), "uint16", full16), "bfloat16x128")
                            for r in T.unroll(ROWS):
                                S.vsts(zero_dq_ub[r, 0], zero_bf16, full16, "NORM_B16")
                            S.vsts(dw_ub[0], S.vdup(T.float32(0.0), "float32", full32), full32, "NORM_B32")
                        T.copy(zero_dq_ub[0:ROWS, 0:D], dQ[q_idx, sid * ROWS : (sid + 1) * ROWS, 0:D])
                        T.copy(dw_ub[0:ROWS], dW[q_idx, sid * ROWS : (sid + 1) * ROWS])
                    else:
                        T.copy(Q[q_idx, 0:H, 0:D], Q_l1)
                        T.copy(Q_l1, a1_l0)
                        T.copy(Q_l1, b3_l0, transpose=True)

                        T.copy(W[q_idx, sid * ROWS : (sid + 1) * ROWS], w_ub[0:ROWS])
                        init_query_state(w_ub, w_scaled_ub, dw_ub, scale)

                        T.set_atomic("add", accum_dtype)

                        for tile_k in T.Pipelined(n_tiles, num_stages=loop_stages):
                            k_off = tile_k * BK
                            stage_gather_and_gemm1(
                                q_idx, k_off, k_base, sid, Indices, dI, K_gm, S_k_val,
                                idx_ub, dI_ub, K_ub, K_l1, a1_l0, b1_l0, A_l0c, A_ub,
                            )
                            stage_vec_and_gemm23(
                                tile_k, S_k_val, scale,
                                A_ub, idx_ub, dI_ub, w_scaled_ub, dw_ub, dA_store_ub, dA_l1, K_l1,
                                a2_l0, b2_l0, dQ_l0c, a3_l0, b3_l0, dK_l0c, dK_ub,
                            )
                            stage_scatter_dk(k_base, sid, S_k_val, idx_ub, dK_ub, dK_fp32)

                        T.set_atomic_none()

                        T.copy(dQ_l0c, dQ[q_idx, 0:H, 0:D])
                        T.copy(dw_ub[0:ROWS], dW[q_idx, sid * ROWS : (sid + 1) * ROWS])

                cur_t = cur_t + 1
                if cur_t == S_q_val:
                    cur_t = 0
                    cur_k_base = cur_k_base + S_k_val

    return main


def build_zero_fp32_kernel(D: int = 128, *, num_cores: int = 32, tile_rows: int = 64):
    """Pure-SIMD TileLang zeroing kernel for ``dk_fp32 [N_k, D]`` (no aclnn OPP dependency)."""
    N_k = T.dynamic("N_k")
    num_aiv = num_cores * 2
    assert D == 128

    @T.prim_func
    def zero_fp32(buf: T.Tensor((N_k, D), "float32")):
        with T.Kernel(num_aiv) as bx:
            ub = T.alloc_shared((tile_rows, D), "float32")
            with T.SimdVF():
                full32 = S.pset(32, "PAT_ALL")
                z = S.vdup(T.float32(0.0), "float32", full32)
                for r in range(tile_rows):
                    S.vsts(ub[r, 0], z, full32, "NORM_B32")
                    S.vsts(ub[r, 64], z, full32, "NORM_B32")

            rows_per_core = N_k // num_aiv
            rem = N_k % num_aiv
            r_start = bx * rows_per_core + T.min(bx, rem)
            r_count = rows_per_core + T.if_then_else(bx < rem, 1, 0)
            n_chunks = T.ceildiv(r_count, tile_rows)
            for c in T.serial(n_chunks):
                off = r_start + c * tile_rows
                valid = T.min(tile_rows, r_start + r_count - off)
                T.copy(ub[0:valid, 0:D], buf[off : off + valid, 0:D])

    return zero_fp32


def build_cast_fp32_to_bf16_kernel(D: int = 128, *, num_cores: int = 32, tile_rows: int = 64):
    """Pure-SIMD TileLang FP32->BF16 cast kernel for ``dk_fp32 -> dk_bf16`` (no aclnn OPP dependency)."""
    N_k = T.dynamic("N_k")
    num_aiv = num_cores * 2
    assert D == 128

    @T.prim_func
    def cast_fp32_to_bf16(
        src: T.Tensor((N_k, D), "float32"),
        dst: T.Tensor((N_k, D), "bfloat16"),
    ):
        with T.Kernel(num_aiv) as bx:
            src_ub = T.alloc_shared((tile_rows, D), "float32")
            dst_ub = T.alloc_shared((tile_rows, D), "bfloat16")

            rows_per_core = N_k // num_aiv
            rem = N_k % num_aiv
            r_start = bx * rows_per_core + T.min(bx, rem)
            r_count = rows_per_core + T.if_then_else(bx < rem, 1, 0)
            n_chunks = T.ceildiv(r_count, tile_rows)

            for c in T.Pipelined(n_chunks, num_stages=2):
                off = r_start + c * tile_rows
                valid = T.min(tile_rows, r_start + r_count - off)
                T.copy(src[off : off + valid, 0:D], src_ub[0:valid, 0:D])
                with T.SimdVF():
                    full32 = S.pset(32, "PAT_ALL")
                    full16 = S.pset(16, "PAT_ALL")
                    for r in range(tile_rows):
                        e, o = S.vld2(src_ub[r, 0], "DINTLV_B32")
                        bf_e = S.vcvt(e, "bfloat16", full32, sat=False, part=0)
                        bf_o = S.vcvt(o, "bfloat16", full32, sat=False, part=1)
                        merged = S.vor(
                            T.reinterpret(bf_e, "uint16x128"),
                            T.reinterpret(bf_o, "uint16x128"),
                            full16,
                        )
                        S.vsts(dst_ub[r, 0], T.reinterpret(merged, "bfloat16x128"), full16, "NORM_B16")
                T.copy(dst_ub[0:valid, 0:D], dst[off : off + valid, 0:D])

    return cast_fp32_to_bf16


_COMPILE_COUNT = 0


def get_compile_count() -> int:
    """Return how many times the cached main kernel builder has compiled a kernel."""
    return _COMPILE_COUNT


def reset_compile_cache() -> None:
    """Clear all compiled kernel caches and reset the compile counter."""
    global _COMPILE_COUNT
    _get_compiled_kernel.cache_clear()
    _get_compiled_zero_kernel.cache_clear()
    _get_compiled_cast_kernel.cache_clear()
    _COMPILE_COUNT = 0


@lru_cache(maxsize=32)
def _get_compiled_kernel(
    H: int,
    D: int,
    K: int,
    BK: int,
    num_cores: int,
    sorted_invalid_tail: bool,
    num_stages: int,
    staged_pipeline: bool,
    interleaved_pipeline: bool,
    dk_l2_policy: str,
    k_l2_policy: str,
):
    global _COMPILE_COUNT
    _COMPILE_COUNT += 1
    prim_func = build_lightning_indexer_sparse_bwd_kernel(
        H=H,
        D=D,
        K=K,
        BK=BK,
        num_cores=num_cores,
        sorted_invalid_tail=sorted_invalid_tail,
        num_stages=num_stages,
        staged_pipeline=staged_pipeline,
        interleaved_pipeline=interleaved_pipeline,
        dk_l2_policy=dk_l2_policy,
        k_l2_policy=k_l2_policy,
    )
    return tilelang.compile(prim_func, target="ascend", out_idx=[])


@lru_cache(maxsize=8)
def _get_compiled_zero_kernel(D: int, num_cores: int):
    return tilelang.compile(build_zero_fp32_kernel(D=D, num_cores=num_cores), target="ascend", out_idx=[])


@lru_cache(maxsize=8)
def _get_compiled_cast_kernel(D: int, num_cores: int):
    return tilelang.compile(build_cast_fp32_to_bf16_kernel(D=D, num_cores=num_cores), target="ascend", out_idx=[])


def lightning_indexer_sparse_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    indices: torch.Tensor,
    dI: torch.Tensor,
    scale: float = 1.0,
    dk_workspace: Optional[torch.Tensor] = None,
    sorted_invalid_tail: bool = True,
    *,
    accumulate_dk: bool = False,
    bk: int = 128,
    num_cores: int = 32,
    num_stages: int = 2,
    staged_pipeline: bool = True,
    interleaved_pipeline: bool = True,
    dk_l2_policy: str = "NORMAL_FV",
    k_l2_policy: str = "NORMAL_FV",
    use_tilelang_aux: bool = True,
    dq_out: Optional[torch.Tensor] = None,
    dk_out: Optional[torch.Tensor] = None,
    dw_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek-V4.1-Flash Sparse Lightning Indexer Backward on Ascend 950.

    Matches the v1 interface in Section 4.1 of ``lightning_indexer_bwd_spec.md``.

    Args:
        q: ``[B, S_q, H, D]``, ``bfloat16``, contiguous (v1 static ``H=32, D=128``).
        k: ``[B, S_k, D]``, ``bfloat16``, contiguous.
        w: ``[B, S_q, H]``, ``float32``, contiguous.
        indices: ``[B, S_q, K]``, ``int32``, contiguous (v1 static ``K=512``, values in ``[0, S_k)`` or ``-1``).
        dI: ``[B, S_q, K]``, ``float32``, contiguous (v1 static ``K=512``).
        scale: Optional dot-product scale ``alpha`` (default ``1.0``).
        dk_workspace: Optional ``[B, S_k, D]`` ``float32`` workspace buffer for atomic ``dK`` accumulation.
        sorted_invalid_tail: If ``True``, enable tile-level early exit when ``indices[b, t, k_start] < 0``.
        accumulate_dk: If ``True`` and ``dk_workspace`` is provided, accumulate into ``dk_workspace``
            without zeroing it first (supports CSA2 Reindex cross-layer ``dK`` accumulation).
        bk: K-tile block size (``64`` or ``128``).
        num_cores: Persistent AI Core count (``28``, ``32`` [acceptance standard], or ``36``).
        num_stages: Software pipeline stage budget for the K-tile loop.
        staged_pipeline: If ``True``, enable 3-stage ``T.Stage`` cross-iteration overlap.
        dk_l2_policy: L2 store cache policy for ``dK_fp32`` atomic scatter (default ``"NORMAL_FV"``).
        use_tilelang_aux: If ``True``, use pure-SIMD TileLang kernels to zero ``dk_workspace``
            and cast ``dk_fp32 -> dk_bf16`` (avoids toolkit-only CANN missing aclnn OPP ops).
        dq_out / dk_out / dw_out: Optional pre-allocated output tensors (for zero-alloc bench / NaN sentinels).

    Returns:
        ``(dq, dk, dw)`` with shapes ``[B, S_q, H, D]`` (``bf16``), ``[B, S_k, D]`` (``bf16``),
        and ``[B, S_q, H]`` (``fp32``).
    """
    if q.ndim != 4:
        raise ValueError(f"Expected q to be 4D [B, S_q, H, D], got {tuple(q.shape)}")
    B, S_q, H, D = q.shape
    if k.ndim != 3 or k.shape[0] != B or k.shape[2] != D:
        raise ValueError(f"Expected k shape [B={B}, S_k, D={D}], got {tuple(k.shape)}")
    S_k = k.shape[1]
    if w.shape != (B, S_q, H):
        raise ValueError(f"Expected w shape {(B, S_q, H)}, got {tuple(w.shape)}")
    if indices.ndim != 3 or indices.shape[:2] != (B, S_q):
        raise ValueError(f"Expected indices shape [B={B}, S_q={S_q}, K], got {tuple(indices.shape)}")
    K_dim = indices.shape[2]
    if dI.shape != (B, S_q, K_dim):
        raise ValueError(f"Expected dI shape {(B, S_q, K_dim)}, got {tuple(dI.shape)}")

    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError(f"v1 requires q/k bfloat16, got q={q.dtype}, k={k.dtype}")
    if w.dtype != torch.float32 or dI.dtype != torch.float32:
        raise TypeError(f"v1 requires w/dI float32, got w={w.dtype}, dI={dI.dtype}")
    if indices.dtype != torch.int32:
        raise TypeError(f"v1 requires indices int32, got {indices.dtype}")

    q = q.contiguous()
    k = k.contiguous()
    w = w.contiguous()
    indices = indices.contiguous()
    dI = dI.contiguous()

    N_q = B * S_q
    N_k = B * S_k

    if dq_out is None:
        dq = torch.empty((B, S_q, H, D), dtype=torch.bfloat16, device=q.device)
    else:
        assert dq_out.shape == (B, S_q, H, D) and dq_out.dtype == torch.bfloat16 and dq_out.is_contiguous()
        dq = dq_out

    if dw_out is None:
        dw = torch.empty((B, S_q, H), dtype=torch.float32, device=q.device)
    else:
        assert dw_out.shape == (B, S_q, H) and dw_out.dtype == torch.float32 and dw_out.is_contiguous()
        dw = dw_out

    if dk_workspace is None:
        if accumulate_dk:
            raise ValueError("accumulate_dk=True requires an explicit dk_workspace tensor")
        dk_fp32 = torch.empty((B, S_k, D), dtype=torch.float32, device=q.device)
        need_zero = True
    else:
        if dk_workspace.shape != (B, S_k, D) or dk_workspace.dtype != torch.float32 or not dk_workspace.is_contiguous():
            raise ValueError(
                f"dk_workspace must be contiguous float32 with shape {(B, S_k, D)}, "
                f"got shape={tuple(dk_workspace.shape)}, dtype={dk_workspace.dtype}, "
                f"contiguous={dk_workspace.is_contiguous()}"
            )
        dk_fp32 = dk_workspace
        need_zero = not accumulate_dk

    if need_zero:
        if use_tilelang_aux:
            _get_compiled_zero_kernel(D, num_cores)(dk_fp32.view(N_k, D))
        else:
            dk_fp32.zero_()

    kernel = _get_compiled_kernel(
        H, D, K_dim, bk, num_cores, sorted_invalid_tail, num_stages,
        staged_pipeline, interleaved_pipeline, dk_l2_policy, k_l2_policy,
    )
    kernel(
        q.view(N_q, H, D),
        k.view(N_k, D),
        w.view(N_q, H),
        indices.view(N_q, K_dim),
        dI.view(N_q, K_dim),
        dq.view(N_q, H, D),
        dk_fp32.view(N_k, D),
        dw.view(N_q, H),
        int(S_q),
        int(S_k),
        float(scale),
    )

    if use_tilelang_aux:
        if dk_out is None:
            dk = torch.empty((B, S_k, D), dtype=torch.bfloat16, device=q.device)
        else:
            assert dk_out.shape == (B, S_k, D) and dk_out.dtype == torch.bfloat16 and dk_out.is_contiguous()
            dk = dk_out
        _get_compiled_cast_kernel(D, num_cores)(dk_fp32.view(N_k, D), dk.view(N_k, D))
    else:
        dk = dk_fp32.to(torch.bfloat16)
        if dk_out is not None:
            dk_out.copy_(dk)
            dk = dk_out

    return dq, dk, dw
