"""Single-Query (Lq=1) Variable-Length Multi-Head Attention Forward for Ascend NPU.

Spec contract:
    out, lse = qsmall_varlen_attn_fwd(q, k, v, cu_seqlens_k, sm_scale)

Fixed compile-time specialization (per spec):
    - H = 56 (MHA, H_q = H_k = H_v = 56)
    - D = 32
    - Q, K, V, Out: bfloat16 (contiguous row-major)
    - LSE: float32 (natural-log scale: m + ln(l); -inf for empty sequences)
    - cu_seqlens_k: int32 (length N + 1)

Runtime dynamic parameters (zero recompilation):
    - N (batch / sample count)
    - total_k (total ragged KV tokens; int64 pointer arithmetic to avoid int32 overflow)
    - seqlen_k[i] = cu_seqlens_k[i + 1] - cu_seqlens_k[i]
    - sm_scale (float32 scalar)
    - grid_n (active AIV core count; supports 950PR 28AIC/56AIV, 950DT 32AIC/64AIV,
      and small-N launches from a single compiled binary)

Microarchitecture (Pure AIV __vector__ + 256B SimdVF, V2):
    - Packs H_PACK=8 consecutive heads per task (8 * 32 * 2B = 512B = exact 1 L2 cacheline
      per token, 7 head-groups per sample, zero head-masking).
    - Unified 3-stage T.Pipelined KV loop (block_n=64, num_stages=3 -> 197.8 KiB / 248 KiB UB)
      eliminating separate tail UB buffers while keeping a branch-free full-block inner loop
      and an OOB-safe tail path with -inf score masking.
    - Split QK+Softmax SimdVF and PV SimdVF for better pipeline overlap, with NOTALLOC_KEEP
      L2 policy to avoid L2 thrashing on single-pass K/V streaming.
    - Even/Odd BF16<->FP32 SIMD lane pairing (PART_EVEN / PART_ODD) packs 4 full heads
      (4 * 16 = 64 FP32 lanes) into one 2048-bit vector register for both QK^T and PV.
    - QK^T per 4 heads: vld -> 2x vcvt -> vmul + vmula -> vcgadd -> vcpadd.
    - Online Softmax over BLOCK_N tokens x 8 heads without transpose using cyclic
      vselr(lane ^ 32/16/8) reductions and native vexpdif (exact natural exp(s - m)).
"""

import math
import os
from typing import Any

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

# Compile-time fixed spec constants
NUM_HEADS = 56
HEAD_DIM = 32
DEFAULT_H_PACK = 8  # 8 * 32 * 2B = 512B (exact 1 L2 cacheline on Ascend 950)
DEFAULT_BLOCK_N = 64  # Must be a positive multiple of 8 (64 tokens x 8 heads = 8 full FP32 vectors)
DEFAULT_NUM_STAGES = 3  # 3 stages at block_n=64 -> 197.8 KiB / 248 KiB AIV UB
DEFAULT_L2_CACHE_CTRL = "NOTALLOC_KEEP"

_KERNEL_CACHE: dict[tuple[int, int, str], Any] = {}
_COMPILE_COUNT = 0


def get_compile_count() -> int:
    """Return total number of TileLang compilations performed in this process."""
    return _COMPILE_COUNT


def clear_kernel_cache() -> None:
    """Clear the in-process compiled kernel cache and reset compile counter."""
    global _COMPILE_COUNT
    _KERNEL_CACHE.clear()
    _COMPILE_COUNT = 0


def detect_num_aiv(default: int = 64, max_aiv: int | None = None) -> int:
    """Detect active AIV core count on the current Ascend device.

    Supports Ascend 950PR (28 AIC -> 56 AIV), 950DT (32 AIC -> 64 AIV,
    36 AIC -> 72 AIV), or explicit override via env var ``QSMALL_ATTN_NUM_AIV``.
    """
    env_val = os.environ.get("QSMALL_ATTN_NUM_AIV")
    if env_val is not None:
        parsed = int(env_val)
        if parsed > 0:
            return min(parsed, max_aiv) if max_aiv is not None else parsed

    detected = default
    npu = getattr(torch, "npu", None)
    if npu is not None:
        try:
            if npu.is_available():
                props = npu.get_device_properties(npu.current_device())
                vec_cores = getattr(props, "vector_core_num", None)
                if isinstance(vec_cores, int) and vec_cores > 0:
                    detected = vec_cores
                else:
                    cube_cores = getattr(props, "cube_core_num", None)
                    if isinstance(cube_cores, int) and cube_cores > 0:
                        detected = cube_cores * 2
                    else:
                        mp_count = getattr(props, "multi_processor_count", None)
                        if isinstance(mp_count, int) and mp_count > 0:
                            detected = mp_count * 2 if mp_count <= 40 else mp_count
        except Exception:
            pass

    return min(detected, max_aiv) if max_aiv is not None else detected


def build_qsmall_varlen_attn_fwd(
    *,
    block_n: int = DEFAULT_BLOCK_N,
    num_stages: int = DEFAULT_NUM_STAGES,
    l2_cache_ctrl: str = DEFAULT_L2_CACHE_CTRL,
):
    """Build the TileLang PrimFunc for qsmall_varlen_attn_fwd.

    The resulting PrimFunc accepts dynamic N, total_k (int64), sm_scale (float32),
    and grid_n (int32) without recompilation.
    """
    if block_n <= 0 or block_n % 8 != 0:
        raise ValueError(f"block_n must be a positive multiple of 8, got {block_n}")
    if num_stages not in (1, 2, 3, 4):
        raise ValueError(f"num_stages must be 1, 2, 3, or 4, got {num_stages}")

    # Static UB capacity guard (Ascend 950 VectorCoreSpec.ub_size = 248 KiB, reserve 8 KiB stack/misc)
    est_ub_bytes = (2 * num_stages) * block_n * DEFAULT_H_PACK * HEAD_DIM * 2 + block_n * DEFAULT_H_PACK * 4 + 4096
    if est_ub_bytes > 245760:
        raise ValueError(
            f"Configuration (block_n={block_n}, num_stages={num_stages}) requires ~{est_ub_bytes / 1024:.1f} KiB UB, "
            "exceeding the 240 KiB safe Ascend 950 AIV UB budget (248 KiB minus 8 KiB reserved). "
            "Use (block_n=64, num_stages=3) or (block_n=48, num_stages=4)."
        )

    H = NUM_HEADS
    D = HEAD_DIM
    H_PACK = DEFAULT_H_PACK
    NUM_H_GROUPS = H // H_PACK  # 7
    BLOCK_N = block_n
    NUM_VEC_S = BLOCK_N // 8  # Each 64-lane FP32 vector holds 8 tokens x 8 heads
    STAGES = num_stages
    L2_CTRL = l2_cache_ctrl

    N = T.dynamic("N", dtype="int32")
    total_k = T.dynamic("total_k", dtype="int64")

    @T.macro
    def _online_softmax_update(
        s_ub,
        m_ub,
        l_ub,
        alpha_ub,
        is_first,
    ):
        """Block-level online softmax on s_ub[BLOCK_N, 8] -> writes P in-place into s_ub."""
        full32 = S.pset(32)
        lane = S.vci(0, "int32", "INC_ORDER")
        idx_rot32 = S.vxor(lane, S.vdup(32, "int32", full32), full32)
        idx_rot16 = S.vxor(lane, S.vdup(16, "int32", full32), full32)
        idx_rot8 = S.vxor(lane, S.vdup(8, "int32", full32), full32)

        m_blk = S.alloc_var("float32")
        m_blk = S.vld(s_ub[0, 0])
        for k_vec in T.unroll(1, NUM_VEC_S):
            m_blk = S.vmax(m_blk, S.vld(s_ub[k_vec * 8, 0]), full32)
        m_blk = S.vmax(m_blk, S.vselr(m_blk, idx_rot32), full32)
        m_blk = S.vmax(m_blk, S.vselr(m_blk, idx_rot16), full32)
        m_blk = S.vmax(m_blk, S.vselr(m_blk, idx_rot8), full32)

        m_cur = S.alloc_var("float32")
        alpha = S.alloc_var("float32")
        if is_first != 0:
            m_cur = m_blk
            alpha = S.vdup(0.0, "float32", full32)
        else:
            m_prev = S.vld(m_ub[0])
            m_cur = S.vmax(m_prev, m_blk, full32)
            alpha = S.vexpdif(m_prev, m_cur, full32)
        S.vsts(m_ub[0], m_cur, full32)
        S.vsts(alpha_ub[0], alpha, full32)

        l_blk = S.alloc_var("float32")
        l_blk = S.vdup(0.0, "float32", full32)
        for k_vec in T.unroll(NUM_VEC_S):
            s_vec = S.vld(s_ub[k_vec * 8, 0])
            p_vec = S.vexpdif(s_vec, m_cur, full32)
            S.vsts(s_ub[k_vec * 8, 0], p_vec, full32)
            l_blk = S.vadd(l_blk, p_vec, full32)

        l_blk = S.vadd(l_blk, S.vselr(l_blk, idx_rot32), full32)
        l_blk = S.vadd(l_blk, S.vselr(l_blk, idx_rot16), full32)
        l_blk = S.vadd(l_blk, S.vselr(l_blk, idx_rot8), full32)

        if is_first != 0:
            S.vsts(l_ub[0], l_blk, full32)
        else:
            l_prev = S.vld(l_ub[0])
            l_new = S.vadd(S.vmul(l_prev, alpha, full32), l_blk, full32)
            S.vsts(l_ub[0], l_new, full32)

        S.mem_bar("VST_VLD")

    @T.prim_func
    def main(
        Q: T.Tensor((N, H, D), "bfloat16"),
        K: T.Tensor((total_k, H, D), "bfloat16"),
        V: T.Tensor((total_k, H, D), "bfloat16"),
        cu_seqlens_k: T.Tensor((N + 1,), "int32"),
        Out: T.Tensor((N, H, D), "bfloat16"),
        LSE: T.Tensor((N, H), "float32"),
        sm_scale: T.float32,
        grid_n: T.int32,
    ):
        with T.Kernel(grid_n) as bx:
            # Multi-buffered streaming KV tiles (unified full + tail -> allows 3 stages at block_n=64!)
            k_ub = T.alloc_shared((BLOCK_N, H_PACK, D), "bfloat16")
            v_ub = T.alloc_shared((BLOCK_N, H_PACK, D), "bfloat16")

            # Small per-task buffers (~5.8 KiB total)
            q_ub = T.alloc_shared((H_PACK, D), "bfloat16")
            q_scaled_ub = T.alloc_shared((4, 64), "float32")
            o_acc_ub = T.alloc_shared((4, 64), "float32")
            m_ub = T.alloc_shared((64,), "float32")
            l_ub = T.alloc_shared((64,), "float32")
            alpha_ub = T.alloc_shared((64,), "float32")
            s_ub = T.alloc_shared((BLOCK_N, H_PACK), "float32")
            out_ub = T.alloc_shared((H_PACK, D), "bfloat16")
            lse_ub = T.alloc_shared((64,), "float32")

            T.annotate_buffer_versions(
                {
                    k_ub: STAGES,
                    v_ub: STAGES,
                    q_ub: 1,
                    q_scaled_ub: 1,
                    o_acc_ub: 1,
                    m_ub: 1,
                    l_ub: 1,
                    alpha_ub: 1,
                    s_ub: 1,
                    out_ub: 1,
                    lse_ub: 1,
                }
            )

            total_tasks = N * NUM_H_GROUPS
            num_iters = (total_tasks + grid_n - 1) // grid_n
            for task_idx in T.serial(num_iters):
                tid = task_idx * grid_n + bx
                if tid < total_tasks:
                    sample_id = tid // NUM_H_GROUPS
                    h_grp = tid % NUM_H_GROUPS
                    h_start = h_grp * H_PACK

                    k_start = T.cast(cu_seqlens_k[sample_id], "int64")
                    k_end = T.cast(cu_seqlens_k[sample_id + 1], "int64")
                    seqlen = T.cast(k_end - k_start, "int32")

                    if seqlen <= 0:
                        # Empty sequence fast path: out = 0.0 (bf16), lse = -inf (fp32)
                        with T.SimdVF():
                            full32 = S.pset(32)
                            full16 = S.pset(16)
                            mask_vl8 = S.pset(32, "PAT_VL8")
                            zero16 = T.reinterpret(S.vdup(T.uint16(0), "uint16", full16), "bfloat16x128")
                            neg_inf = S.vdup(float("-inf"), "float32", full32)
                            S.vsts(out_ub[0, 0], zero16, full16, "NORM_B16")
                            S.vsts(out_ub[4, 0], zero16, full16, "NORM_B16")
                            S.vsts(lse_ub[0], neg_inf, mask_vl8, extent=8)
                        T.copy(out_ub, Out[sample_id, h_start : h_start + H_PACK, 0:D])
                        T.copy(lse_ub[0:H_PACK], LSE[sample_id, h_start : h_start + H_PACK])
                    else:
                        # Load Q[sample_id, h_start:h_start+8, 0:32] and pre-scale to FP32 even/odd
                        T.copy(Q[sample_id, h_start : h_start + H_PACK, 0:D], q_ub)
                        with T.SimdVF():
                            full32 = S.pset(32)
                            full16 = S.pset(16)
                            scale_vec = S.vdup(sm_scale, "float32", full32)
                            q_raw0 = S.vld(q_ub[0, 0])
                            q_raw1 = S.vld(q_ub[4, 0])
                            qe0 = S.vmul(S.vcvt(q_raw0, "float32", full16, part=0), scale_vec, full32)
                            qo0 = S.vmul(S.vcvt(q_raw0, "float32", full16, part=1), scale_vec, full32)
                            qe1 = S.vmul(S.vcvt(q_raw1, "float32", full16, part=0), scale_vec, full32)
                            qo1 = S.vmul(S.vcvt(q_raw1, "float32", full16, part=1), scale_vec, full32)
                            S.vsts(q_scaled_ub[0, 0], qe0, full32)
                            S.vsts(q_scaled_ub[1, 0], qo0, full32)
                            S.vsts(q_scaled_ub[2, 0], qe1, full32)
                            S.vsts(q_scaled_ub[3, 0], qo1, full32)

                        num_blocks = (seqlen + BLOCK_N - 1) // BLOCK_N

                        # ---------------------------------------------------------
                        # Unified pipelined KV loop (3-stage MTE2 <-> V overlap)
                        # ---------------------------------------------------------
                        for blk in T.Pipelined(num_blocks, num_stages=STAGES):
                            base = k_start + T.cast(blk * BLOCK_N, "int64")
                            valid = T.min(BLOCK_N, seqlen - blk * BLOCK_N)
                            T.copy(
                                K[base : base + valid, h_start : h_start + H_PACK, 0:D],
                                k_ub[0:valid, :, :],
                                l2_cache_ctrl=L2_CTRL,
                            )
                            T.copy(
                                V[base : base + valid, h_start : h_start + H_PACK, 0:D],
                                v_ub[0:valid, :, :],
                                l2_cache_ctrl=L2_CTRL,
                            )

                            # VF 1: QK^T + Online Softmax
                            with T.SimdVF():
                                full32 = S.pset(32)
                                full16 = S.pset(16)
                                mask_vl4 = S.pset(32, "PAT_VL4")
                                mask_vl8 = S.pset(32, "PAT_VL8")
                                lane = S.vci(0, "int32", "INC_ORDER")
                                idx_shift4 = S.vand(
                                    S.vadds(lane, -4, full32),
                                    S.vdup(63, "int32", full32),
                                    full32,
                                )

                                if valid < BLOCK_N:
                                    neg_inf = S.vdup(float("-inf"), "float32", full32)
                                    for kv_i in T.unroll(NUM_VEC_S):
                                        S.vsts(s_ub[kv_i * 8, 0], neg_inf, full32)
                                    S.mem_bar("VST_VLD")

                                qe0 = S.vld(q_scaled_ub[0, 0])
                                qo0 = S.vld(q_scaled_ub[1, 0])
                                qe1 = S.vld(q_scaled_ub[2, 0])
                                qo1 = S.vld(q_scaled_ub[3, 0])

                                prod0 = S.alloc_var("float32")
                                prod1 = S.alloc_var("float32")
                                if valid == BLOCK_N:
                                    for j in T.serial(BLOCK_N):
                                        k_raw0 = S.vld(k_ub[j, 0, 0])
                                        k_raw1 = S.vld(k_ub[j, 4, 0])
                                        ke0 = S.vcvt(k_raw0, "float32", full16, part=0)
                                        ko0 = S.vcvt(k_raw0, "float32", full16, part=1)
                                        ke1 = S.vcvt(k_raw1, "float32", full16, part=0)
                                        ko1 = S.vcvt(k_raw1, "float32", full16, part=1)

                                        prod0 = S.vmul(qe0, ke0, full32)
                                        S.vmula(prod0, qo0, ko0, full32)
                                        s_g0 = S.vcpadd(S.vcgadd(prod0, full32), full32)

                                        prod1 = S.vmul(qe1, ke1, full32)
                                        S.vmula(prod1, qo1, ko1, full32)
                                        s_g1 = S.vcpadd(S.vcgadd(prod1, full32), full32)

                                        s8 = S.vsel(s_g0, S.vselr(s_g1, idx_shift4), mask_vl4)
                                        S.vsts(s_ub[j, 0], s8, mask_vl8, extent=8)
                                else:
                                    for j in T.serial(valid):
                                        k_raw0 = S.vld(k_ub[j, 0, 0])
                                        k_raw1 = S.vld(k_ub[j, 4, 0])
                                        ke0 = S.vcvt(k_raw0, "float32", full16, part=0)
                                        ko0 = S.vcvt(k_raw0, "float32", full16, part=1)
                                        ke1 = S.vcvt(k_raw1, "float32", full16, part=0)
                                        ko1 = S.vcvt(k_raw1, "float32", full16, part=1)

                                        prod0 = S.vmul(qe0, ke0, full32)
                                        S.vmula(prod0, qo0, ko0, full32)
                                        s_g0 = S.vcpadd(S.vcgadd(prod0, full32), full32)

                                        prod1 = S.vmul(qe1, ke1, full32)
                                        S.vmula(prod1, qo1, ko1, full32)
                                        s_g1 = S.vcpadd(S.vcgadd(prod1, full32), full32)

                                        s8 = S.vsel(s_g0, S.vselr(s_g1, idx_shift4), mask_vl4)
                                        S.vsts(s_ub[j, 0], s8, mask_vl8, extent=8)

                                S.mem_bar("VST_VLD")

                                is_first_blk = T.if_then_else(blk == 0, 1, 0)
                                _online_softmax_update(s_ub, m_ub, l_ub, alpha_ub, is_first_blk)

                            # VF 2: PV tile accumulation + O_acc rescale
                            with T.SimdVF():
                                full32 = S.pset(32)
                                full16 = S.pset(16)
                                lane = S.vci(0, "int32", "INC_ORDER")
                                idx_brc_g0 = S.vshrs(lane, 4, full32)
                                idx_brc_g1 = S.vadds(idx_brc_g0, 4, full32)

                                o_tile_e0 = S.alloc_var("float32")
                                o_tile_o0 = S.alloc_var("float32")
                                o_tile_e1 = S.alloc_var("float32")
                                o_tile_o1 = S.alloc_var("float32")
                                o_tile_e0 = S.vdup(0.0, "float32", full32)
                                o_tile_o0 = S.vdup(0.0, "float32", full32)
                                o_tile_e1 = S.vdup(0.0, "float32", full32)
                                o_tile_o1 = S.vdup(0.0, "float32", full32)

                                if valid == BLOCK_N:
                                    for k_vec in T.serial(NUM_VEC_S):
                                        p_k = S.vld(s_ub[k_vec * 8, 0])
                                        for t in T.unroll(8):
                                            j_tok = k_vec * 8 + t
                                            v_raw0 = S.vld(v_ub[j_tok, 0, 0])
                                            v_raw1 = S.vld(v_ub[j_tok, 4, 0])
                                            ve0 = S.vcvt(v_raw0, "float32", full16, part=0)
                                            vo0 = S.vcvt(v_raw0, "float32", full16, part=1)
                                            ve1 = S.vcvt(v_raw1, "float32", full16, part=0)
                                            vo1 = S.vcvt(v_raw1, "float32", full16, part=1)

                                            p_g0 = S.vselr(p_k, S.vadds(idx_brc_g0, t * 8, full32))
                                            p_g1 = S.vselr(p_k, S.vadds(idx_brc_g1, t * 8, full32))

                                            S.vmula(o_tile_e0, p_g0, ve0, full32)
                                            S.vmula(o_tile_o0, p_g0, vo0, full32)
                                            S.vmula(o_tile_e1, p_g1, ve1, full32)
                                            S.vmula(o_tile_o1, p_g1, vo1, full32)
                                else:
                                    num_tail_vecs = (valid + 7) // 8
                                    for k_vec in T.serial(num_tail_vecs):
                                        p_k = S.vld(s_ub[k_vec * 8, 0])
                                        sub_rem = T.min(8, valid - k_vec * 8)
                                        for t in T.serial(sub_rem):
                                            j_tok = k_vec * 8 + t
                                            v_raw0 = S.vld(v_ub[j_tok, 0, 0])
                                            v_raw1 = S.vld(v_ub[j_tok, 4, 0])
                                            ve0 = S.vcvt(v_raw0, "float32", full16, part=0)
                                            vo0 = S.vcvt(v_raw0, "float32", full16, part=1)
                                            ve1 = S.vcvt(v_raw1, "float32", full16, part=0)
                                            vo1 = S.vcvt(v_raw1, "float32", full16, part=1)

                                            p_g0 = S.vselr(p_k, S.vadds(idx_brc_g0, t * 8, full32))
                                            p_g1 = S.vselr(p_k, S.vadds(idx_brc_g1, t * 8, full32))

                                            S.vmula(o_tile_e0, p_g0, ve0, full32)
                                            S.vmula(o_tile_o0, p_g0, vo0, full32)
                                            S.vmula(o_tile_e1, p_g1, ve1, full32)
                                            S.vmula(o_tile_o1, p_g1, vo1, full32)

                                is_first_o = T.if_then_else(blk == 0, 1, 0)
                                if is_first_o != 0:
                                    S.vsts(o_acc_ub[0, 0], o_tile_e0, full32)
                                    S.vsts(o_acc_ub[1, 0], o_tile_o0, full32)
                                    S.vsts(o_acc_ub[2, 0], o_tile_e1, full32)
                                    S.vsts(o_acc_ub[3, 0], o_tile_o1, full32)
                                else:
                                    alpha_rep = S.vld(alpha_ub[0])
                                    alpha_g0 = S.vselr(alpha_rep, idx_brc_g0)
                                    alpha_g1 = S.vselr(alpha_rep, idx_brc_g1)

                                    oe0 = S.alloc_var("float32")
                                    oo0 = S.alloc_var("float32")
                                    oe1 = S.alloc_var("float32")
                                    oo1 = S.alloc_var("float32")
                                    oe0 = S.vld(o_acc_ub[0, 0])
                                    oo0 = S.vld(o_acc_ub[1, 0])
                                    oe1 = S.vld(o_acc_ub[2, 0])
                                    oo1 = S.vld(o_acc_ub[3, 0])

                                    S.vmadd(oe0, alpha_g0, o_tile_e0, full32)
                                    S.vmadd(oo0, alpha_g0, o_tile_o0, full32)
                                    S.vmadd(oe1, alpha_g1, o_tile_e1, full32)
                                    S.vmadd(oo1, alpha_g1, o_tile_o1, full32)

                                    S.vsts(o_acc_ub[0, 0], oe0, full32)
                                    S.vsts(o_acc_ub[1, 0], oo0, full32)
                                    S.vsts(o_acc_ub[2, 0], oe1, full32)
                                    S.vsts(o_acc_ub[3, 0], oo1, full32)

                        # ---------------------------------------------------------
                        # Epilogue: normalize O -> BF16, compute LSE = m + ln(l)
                        # ---------------------------------------------------------
                        with T.SimdVF():
                            full32 = S.pset(32)
                            full16 = S.pset(16)
                            mask_vl8 = S.pset(32, "PAT_VL8")

                            lane = S.vci(0, "int32", "INC_ORDER")
                            idx_brc_g0 = S.vshrs(lane, 4, full32)
                            idx_brc_g1 = S.vadds(idx_brc_g0, 4, full32)

                            m_final = S.vld(m_ub[0])
                            l_final = S.vld(l_ub[0])
                            inv_l = S.vdiv(S.vdup(1.0, "float32", full32), l_final, full32)
                            lse_vec = S.vadd(m_final, S.vln(l_final, full32), full32)
                            S.vsts(lse_ub[0], lse_vec, mask_vl8, extent=8)

                            inv_l_g0 = S.vselr(inv_l, idx_brc_g0)
                            inv_l_g1 = S.vselr(inv_l, idx_brc_g1)

                            out_e0 = S.vmul(S.vld(o_acc_ub[0, 0]), inv_l_g0, full32)
                            out_o0 = S.vmul(S.vld(o_acc_ub[1, 0]), inv_l_g0, full32)
                            out_e1 = S.vmul(S.vld(o_acc_ub[2, 0]), inv_l_g1, full32)
                            out_o1 = S.vmul(S.vld(o_acc_ub[3, 0]), inv_l_g1, full32)

                            bf16_e0 = S.vcvt(out_e0, "bfloat16", full32, sat=False, part=0)
                            bf16_o0 = S.vcvt(out_o0, "bfloat16", full32, sat=False, part=1)
                            merged0 = T.reinterpret(
                                S.vor(
                                    T.reinterpret(bf16_e0, "uint16x128"),
                                    T.reinterpret(bf16_o0, "uint16x128"),
                                    full16,
                                ),
                                "bfloat16x128",
                            )
                            S.vsts(out_ub[0, 0], merged0, full16, "NORM_B16")

                            bf16_e1 = S.vcvt(out_e1, "bfloat16", full32, sat=False, part=0)
                            bf16_o1 = S.vcvt(out_o1, "bfloat16", full32, sat=False, part=1)
                            merged1 = T.reinterpret(
                                S.vor(
                                    T.reinterpret(bf16_e1, "uint16x128"),
                                    T.reinterpret(bf16_o1, "uint16x128"),
                                    full16,
                                ),
                                "bfloat16x128",
                            )
                            S.vsts(out_ub[4, 0], merged1, full16, "NORM_B16")

                        T.copy(out_ub, Out[sample_id, h_start : h_start + H_PACK, 0:D])
                        T.copy(lse_ub[0:H_PACK], LSE[sample_id, h_start : h_start + H_PACK])

    return main


def get_compiled_kernel(
    *,
    block_n: int = DEFAULT_BLOCK_N,
    num_stages: int = DEFAULT_NUM_STAGES,
    l2_cache_ctrl: str = DEFAULT_L2_CACHE_CTRL,
):
    """Compile (once per tuning config) and cache the Ascend kernel."""
    global _COMPILE_COUNT
    key = (block_n, num_stages, l2_cache_ctrl)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        prim_func = build_qsmall_varlen_attn_fwd(
            block_n=block_n,
            num_stages=num_stages,
            l2_cache_ctrl=l2_cache_ctrl,
        )
        kernel = tilelang.compile(prim_func, target="ascend")
        _KERNEL_CACHE[key] = kernel
        _COMPILE_COUNT += 1
    return kernel


def qsmall_varlen_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    sm_scale: float | None = None,
    *,
    num_aiv: int | None = None,
    block_n: int = DEFAULT_BLOCK_N,
    num_stages: int = DEFAULT_NUM_STAGES,
    l2_cache_ctrl: str = DEFAULT_L2_CACHE_CTRL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Single-Query (Lq=1) Varlen Multi-Head Attention Forward on Ascend NPU.

    Args:
        q: [N, 56, 32] bfloat16 contiguous tensor.
        k: [total_k, 56, 32] bfloat16 contiguous tensor.
        v: [total_k, 56, 32] bfloat16 contiguous tensor.
        cu_seqlens_k: [N + 1] int32 contiguous tensor.
        sm_scale: Optional float32 softmax scale (defaults to 1 / sqrt(32)).
        num_aiv: Optional AIV launch count (defaults to auto-detected device AIV count).
        block_n: KV tile size (default 64).
        num_stages: Pipeline buffer stages for K/V GM->UB copies (default 3).
        l2_cache_ctrl: Ascend L2 cache policy for streaming K/V loads.

    Returns:
        (out, lse):
            out: [N, 56, 32] bfloat16 contiguous tensor.
            lse: [N, 56] float32 contiguous tensor (natural-log scale).
    """
    if q.ndim != 3 or q.shape[1] != NUM_HEADS or q.shape[2] != HEAD_DIM:
        raise ValueError(f"Expected q shape [N, {NUM_HEADS}, {HEAD_DIM}], got {tuple(q.shape)}")
    if k.ndim != 3 or k.shape[1] != NUM_HEADS or k.shape[2] != HEAD_DIM:
        raise ValueError(f"Expected k shape [total_k, {NUM_HEADS}, {HEAD_DIM}], got {tuple(k.shape)}")
    if v.shape != k.shape:
        raise ValueError(f"Expected v shape {tuple(k.shape)}, got {tuple(v.shape)}")
    if cu_seqlens_k.ndim != 1 or cu_seqlens_k.shape[0] != q.shape[0] + 1:
        raise ValueError(f"Expected cu_seqlens_k shape [{q.shape[0] + 1}], got {tuple(cu_seqlens_k.shape)}")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16 for q/k/v, got {q.dtype}/{k.dtype}/{v.dtype}")
    if cu_seqlens_k.dtype != torch.int32:
        raise ValueError(f"Expected int32 for cu_seqlens_k, got {cu_seqlens_k.dtype}")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and cu_seqlens_k.is_contiguous()):
        raise ValueError("All input tensors must be contiguous row-major")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    n_samples = q.shape[0]
    out = torch.empty_like(q)
    lse = torch.empty((n_samples, NUM_HEADS), dtype=torch.float32, device=q.device)

    if n_samples == 0:
        return out, lse

    active_aiv = detect_num_aiv() if num_aiv is None else int(num_aiv)
    total_tasks = n_samples * (NUM_HEADS // DEFAULT_H_PACK)
    grid_n = max(1, min(active_aiv, total_tasks))

    kernel = get_compiled_kernel(
        block_n=block_n,
        num_stages=num_stages,
        l2_cache_ctrl=l2_cache_ctrl,
    )
    kernel(q, k, v, cu_seqlens_k, out, lse, float(sm_scale), int(grid_n))
    return out, lse
