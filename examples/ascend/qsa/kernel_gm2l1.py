"""Qwen sparse attention (QSA) grouped-query forward on Ascend NPU.

24 query heads share 2 KV heads, head_dim = 256. BF16 in/out, FP32
accumulation and online-softmax state.

Cube-side gather variant of qsa.py: K/V never touch the AIV Unified
Buffer. Each 4-token quad is fetched by one dynamic-base nd2nz GM->L1 DMA
issued on the Cube, landing straight in the matching NZ fractal slots of
the shared KV L1 buffer (within a fractal the n dim is physically linear,
one 32 B slot per row, so 4 rows starting at slot 4i fill rows r0..r0+3
exactly; the final L1 layout is identical to the old UB relay's). The
quad base indices are read as GM scalars on the Cube (bisheng forbids
__cbuf__ pointer dereference, so L1 cannot serve scalar loads), mirroring
the production GBSA A5 gather (GetValue + DataCopy gm2l1_nd2nz). The old
relay (gm2ub 4-row burst + UB nd2nz
scatter + ub2l1, the ~49.7% AIV mte2 load in decode) is gone; that
traffic now rides the mostly idle AIC mte2 pipe.

Index contract, per query (shared across KV heads):
- top_k slots; valid slots are whole 4-token blocks: slots 4i..4i+3 hold
  four consecutive tokens; only the last run of a query may be short.
- Invalid slots hold -1 in Indices and 0 in Valid. The host passes
  safe = indices.clamp_min(0), so invalid quads gather token 0 and the
  pen mask (valid - 1) * 1e30 drops their softmax contribution.
- Ascending order is allowed (better L2 reuse); 4-alignment is not.

Q is pre-grouped per KV head and zero-padded to MP = 32 rows so each AIV
owns one 16-row cube fractal (see pad_q_heads); O keeps that layout and
the host slices the first group rows per head. top_k must be a multiple
of 128. With dynamic_shape=True, num_tokens and kv_length are symbolic
(T.dynamic) and one compiled kernel serves all shapes.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout
from tilelang.profiler import do_bench
from tvm import tirx


def _copy_kv_rows(dst, src, n_rows, row_stride, row_bytes):
    """Gather n_rows consecutive BF16 KV rows in one DMA (AIV path)."""
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.ascend_copy_gm_to_ubuf"),
        dst,
        src,
        0,
        n_rows,
        row_bytes,
        0,
        0,
        0,
        0,
        row_stride * row_bytes,
        row_bytes,
    )


NUM_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
# Padded group rows = the gemm M dimension. 32 = the proven baseline
# (2 AIV subcores x 16 rows via the hardware L0C->UB dual_copy M-split).
# 16 = the GBSA-parity experiment: group is 12, so 32 pads 2.67x and
# doubles every gemm MAC / softmax row / O merge row versus the
# fractal-minimal 16. Compact NZ supports any row count and the
# dual_copy 2:1 split only needs exact halving, so MP=16 (ROWS=8 per
# subcore) should be expressible end to end.
PAD_M = 32
BLOCK_I = 128

# Host-side zero rows appended to K/V so the cube-side quad DMA
# KV[base:base+4] is always in-bounds: safe bases are clamped to
# [0, total_kv-1], so a quad reads at most 3 rows past the logical end
# (short tail run + invalid slots). Pad rows are zeros and invalid slots
# are softmax-masked, so out-of-logical-range reads contribute exactly
# nothing. This keeps the gather branch-free (see gather_kv).
KV_PAD_ROWS = 3

# Bump when the scheduling-knob surface changes; qsa_tune.py refuses to run
# against a stale copy (the server and this repo are synced by hand).
# v3: always-merge restructure (O_tmp regular edge fix).
# v4: A/B knobs -- aiv path restores the original conditional merge;
#     cube path batches + dedupes the quad-base scalar reads.
# v5: K/V zero padding (KV_PAD_ROWS) makes the cube gather fully
#     unconditional -- v4's 32 unrolled per-quad conditionals could leave
#     the KV_l1 arrive flag inside a never-executed branch (device
#     deadlock) and bloated the sync analysis path space.
TUNE_API_VERSION = 5


def qsa_fwd(
    total_q,
    total_kv,
    top_k=2176,
    sm_scale=None,
    *,
    dynamic_shape=False,
    version_overrides=None,
    num_stages=None,
    cube_gather=True,
    chunk_rows=None,
    pad_m=None,
):
    """Build the QSA forward kernel. total_q / total_kv are num_tokens and
    kv_length; dynamic_shape=True makes them symbolic dimensions.

    version_overrides: optional {"buffer_name": count} retuning the buffer
    version annotations. InsertSync drops a cross-iteration sync point whose
    scheduled distance differs from the buffer's version count, so the counts
    are schedule-shape-dependent tuning knobs (matrix search: qsa_tune.py).

    num_stages: override the T.Pipelined stage count (default 2 when NI > 2
    else 1). Empirically the NI <= 2 sequential shape (stages=1) races with
    the Cube-side gather while stages=2 verifies clean, so this knob exists
    to pipeline small-NI shapes too.

    Both knobs are resolved in plain Python below on purpose: the prim_func
    body is AST-interpreted by TileLang, whose `or`/`and` rewrite coerces
    non-TIR operands to bool (builder.unwrap_cond), so `(d or {})` inside
    the kernel body would evaluate to a bool and crash on `.items()`."""
    if top_k <= 0 or top_k % BLOCK_I:
        raise ValueError(f"top_k must be a positive multiple of {BLOCK_I}")
    MP = PAD_M if pad_m is None else pad_m
    padded_kv = total_kv + KV_PAD_ROWS
    if dynamic_shape:
        total_q = T.dynamic("num_tokens")
        padded_kv = T.dynamic("padded_kv_length")
        total_kv = padded_kv - KV_PAD_ROWS
    D = HEAD_DIM
    BI = BLOCK_I
    VL = 64
    ROWS = MP // 2
    NI = top_k // BI
    ROWS_PER_AIV = BI // 2
    # AIV-relay chunking (matches the original qsa.py): 32-row chunks halve
    # the scatter/ub2l1 flag traffic, but with NI <= 2 the loop runs
    # sequentially and 32-row scatters hit a precision bug, so use 16.
    # chunk_rows knob: 64 halves the nz-scatter/ub2l1 count again (136 -> 68
    # per task) with better DMA efficiency; it needs ~+65KB UB per subcore
    # for the KV staging, which only fits now that M=16 freed the O/S/P
    # buffers (at M=32 it overflows -- that is why the original chose 32).
    CHUNK_ROWS = BI // 4 if NI > 2 else 16
    if chunk_rows is not None and NI > 2:
        # NI <= 2 (e.g. the correctness-small scenario) keeps its own
        # 16-row chunking -- sequential 32-row scatters hit a precision
        # bug -- so the override silently does not apply there.
        if chunk_rows % 4 or not 4 <= chunk_rows <= ROWS_PER_AIV:
            raise ValueError(
                f"chunk_rows must be a multiple of 4 in [4, {ROWS_PER_AIV}], got {chunk_rows}"
            )
        CHUNK_ROWS = chunk_rows
    total_tasks = total_q * NUM_KV_HEADS
    # `for x in range(n)` inside a prim_func builds a TIR loop whose
    # induction variable is a Var, which cannot index a Python list. The
    # builder's ctx_for natively expands any other Python iterable at
    # trace time, so iterate this tuple instead to get Python-int quads
    # for the bases-list indexing.
    quad_ids = tuple(range(BI // 4))
    dtype = "bfloat16"
    accum_dtype = "float32"
    scale = sm_scale if sm_scale is not None else D**-0.5
    # K and V share one L1 buffer (disjoint lifetimes); 3 versions decouple
    # the next gather from current consumers.
    KV_L1_VERSIONS = 3 if NI > 1 else 1
    # Scheduling knobs, resolved in plain Python (see docstring). The
    # prim_func body may only reference them via plain names/subscripts.
    stages = 2 if NI > 2 else 1
    if num_stages is not None:
        if not isinstance(num_stages, int) or num_stages < 1:
            raise ValueError(f"num_stages must be a positive integer, got {num_stages!r}")
        stages = max(1, min(num_stages, NI))
    versions = {
        "KV_l1": KV_L1_VERSIONS,
        "P_l1": 2,
        "qk_l0c": 2,
        "pv_l0c": 2,
        "S_ub": 2,
        "P_nz_ub": 2,
        "val_ub": 2,
        "O_ub": 1,
        # Always-merge (both gather paths): O_tmp gets a regular
        # every-iteration FIX->V edge, so it takes 2 versions like S_ub.
        # The conditional merge (direct write at iteration 0) is
        # schedule-shape-coupled and raced again at M=16 (the aiv restore
        # was never re-validated); always-merge is the structure proven at
        # M=32-aiv (v3) and M=16-cube (v5).
        "O_tmp_ub": 2,
        # AIV-relay staging (original qsa.py values).
        "KV_ub": 2,
        "KV_nz_ub": 2,
        "idx_ub": 2,
    }
    if version_overrides is not None:
        for name, count in version_overrides.items():
            if name not in versions:
                raise ValueError(
                    f"unknown buffer name {name!r}; expected one of {sorted(versions)}"
                )
            versions[name] = count

    @T.macro
    def gather_kv(qk_head, bases, KV, KV_l1):
        # One quad = 4 consecutive tokens (block contract), so one
        # dynamic-base nd2nz GM->L1 DMA covers it; the DMA writes the four
        # rows into fractal slots r0..r0+3 directly. bases holds the 32
        # pre-loaded quad-base scalars (shared between the K and V gathers,
        # deduping 1088 -> 544 scalar GM reads per task). Iterating the
        # quad_ids tuple (not range) makes this a trace-time unrolled
        # Python loop, so quad is a Python int and can index the bases
        # list; a range loop would build a TIR Var and fail.
        #
        # Fully unconditional: K/V carry KV_PAD_ROWS zero rows on the
        # host, so base..base+3 is always in-bounds even for a short tail
        # run; invalid slots are softmax-masked and pad rows are zero, so
        # both contribute exactly nothing. No branches also keeps
        # InsertSync's placement trivially correct -- v4's 32 per-quad
        # conditionals could leave the KV_l1 arrive flag inside a branch
        # that never executed, spinning the gemm's wait forever.
        for quad in quad_ids:
            r = 4 * quad
            base = bases[quad]
            T.copy(KV[base : base + 4, qk_head, :], KV_l1[r : r + 4, :])

    @T.macro
    def gather_kv_aiv(qk_head, sid, KV, indices_ub, KV_ub, KV_nz_ub, KV_l1):
        # Legacy relay (original qsa.py): AIV pulls each quad with a 4-row
        # gm2ub burst, rescatters to NZ in the UB, then ub2l1. Kept as the
        # B-side of the A/B gather-path experiment.
        for chunk in T.Serial(ROWS_PER_AIV // CHUNK_ROWS):
            row_start = chunk * CHUNK_ROWS
            for quad in T.Serial(CHUNK_ROWS // 4):
                r = row_start + 4 * quad
                base = T.int32(indices_ub[sid * ROWS_PER_AIV + r])
                T.assume(base >= 0)
                T.assume(base < total_kv)
                if base + 4 <= total_kv:
                    _copy_kv_rows(
                        T.access_ptr(KV_ub[4 * quad, 0], "w"),
                        T.access_ptr(KV[base, qk_head, 0], "r"),
                        4,
                        NUM_KV_HEADS,
                        D * 2,
                    )
                else:
                    # Tail quad crosses the KV boundary; gather per row.
                    for i in T.Serial(4):
                        idx = T.int32(indices_ub[sid * ROWS_PER_AIV + r + i])
                        T.assume(idx >= 0)
                        T.assume(idx < total_kv)
                        T.copy(KV[idx, qk_head, :], KV_ub[4 * quad + i, :])
            T.copy(KV_ub, KV_nz_ub[:CHUNK_ROWS, :])
            l1_row = sid * ROWS_PER_AIV + row_start
            T.copy(KV_nz_ub[:CHUNK_ROWS, :], KV_l1[l1_row : l1_row + CHUNK_ROWS, :])

    @T.macro
    def softmax_pack(S_ub, Val_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub):
        # Column mask pen = (valid - 1) * 1e30: valid -> 0, invalid -> -1e30.
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            one = S.pset(32, "PAT_VL1")
            b16 = S.pset(16, "PAT_VL128")
            v_lo = S.vcvt(S.vld(Val_ub[0], "NORM"), "float32", full)
            v_hi = S.vcvt(S.vld(Val_ub[VL], "NORM"), "float32", full)
            pen_lo = S.vmuls(S.vadds(v_lo, T.float32(-1.0), full), T.float32(1e30), full)
            pen_hi = S.vmuls(S.vadds(v_hi, T.float32(-1.0), full), T.float32(1e30), full)

            p_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", ROWS + 1, BI),
                dtype,
            )
            nz_stride = T.int32(((ROWS + 1) << 16) | 1)

            for r in range(ROWS):
                s0 = S.vmuls(S.vld(S_ub[r, 0], "NORM"), T.float32(scale), full)
                s1 = S.vmuls(S.vld(S_ub[r, VL], "NORM"), T.float32(scale), full)
                s0 = S.vadd(s0, pen_lo, full)
                s1 = S.vadd(s1, pen_hi, full)
                S.vsts(S_ub[r, 0], s0, full, "NORM_B32")
                S.vsts(S_ub[r, VL], s1, full, "NORM_B32")
                m_all = S.vmax(s0, s1, full)
                # Always fold in the running max: at iteration 0 old_m is
                # the -2^30 init, so this is the identity for any real
                # score and keeps the m/alpha/l state edges regular
                # (present every iteration), which InsertSync requires to
                # build a well-formed rotating flag ring.
                old = S.vld(m_ub[r], "BRC_B32")
                S.vsts(old_m_ub[r], old, one, "ONEPT_B32")
                m_all = S.vmax(m_all, old, full)
                mrow = S.vcmax(m_all, full)
                S.vsts(m_ub[r], mrow, one, "ONEPT_B32")

            S.mem_bar("VST_VLD")

            for r in range(ROWS):
                mrow = S.vld(m_ub[r], "BRC_B32")
                e0, e1 = S.vld2(S_ub[r, 0], "DINTLV_B32")
                p0 = S.vexpdif(e0, mrow, full)
                p1 = S.vexpdif(e1, mrow, full)
                sumall = S.vadd(p0, p1, full)
                b0 = S.vcvt(p0, dtype, full, sat=False, part=0)
                b1 = S.vcvt(p1, dtype, full, sat=False, part=1)
                packed = S.vor(
                    T.reinterpret(b0, "uint16x128"),
                    T.reinterpret(b1, "uint16x128"),
                    b16,
                )
                p_nz_ptr = S.vsstb(
                    T.reinterpret(packed, "bfloat16x128"),
                    p_nz_ptr,
                    nz_stride,
                    b16,
                    update=True,
                )
                psum = S.vcadd(sumall, full)
                # alpha = exp(old_m - m): underflows to exact 0 at
                # iteration 0 for any real score (old_m = -2^30), and is
                # exactly 1 for fully-invalid queries (m stays -2^30), so
                # l = 0*alpha + psum = psum and O = 0*alpha + tmp = tmp
                # there -- bitwise identical to the fresh-start path.
                alpha = S.vexpdif(S.vld(old_m_ub[r], "BRC_B32"), mrow, full)
                S.vsts(alpha_ub[r], alpha, one, "ONEPT_B32")
                lnew = S.vadd(
                    S.vmul(S.vld(l_ub[r], "BRC_B32"), alpha, full),
                    psum,
                    one,
                )
                S.vsts(l_ub[r], lnew, one, "ONEPT_B32")

    @T.prim_func
    def main(
        Q: T.Buffer((total_q, NUM_KV_HEADS, MP, D), dtype),
        K: T.Buffer((padded_kv, NUM_KV_HEADS, D), dtype),
        V: T.Buffer((padded_kv, NUM_KV_HEADS, D), dtype),
        Indices: T.Buffer((total_q, top_k), "int32"),
        Valid: T.Buffer((total_q, top_k), "int32"),
        O: T.Buffer((total_q, NUM_KV_HEADS, MP, D), dtype),
    ):
        with T.MixedKernel(total_tasks) as (tid, sid):
            grp = tid // NUM_KV_HEADS
            g = tid % NUM_KV_HEADS

            Q_l1 = T.alloc_l1((MP, D), dtype)
            KV_l1 = T.alloc_l1((BI, D), dtype)
            P_l1 = T.alloc_l1((MP, BI), dtype)
            qk_l0c = T.alloc_l0c((MP, BI), accum_dtype)
            pv_l0c = T.alloc_l0c((MP, D), accum_dtype)

            S_ub = T.alloc_shared((ROWS, BI), accum_dtype)
            P_nz_ub = T.alloc_shared((ROWS + 1, BI), dtype)
            if not cube_gather:
                # AIV-relay staging (original qsa.py layout).
                KV_ub = T.alloc_shared((CHUNK_ROWS, D), dtype)
                KV_nz_ub = T.alloc_shared((CHUNK_ROWS + 1, D), dtype)
                idx_ub = T.alloc_shared((BI,), "int32")
            val_ub = T.alloc_shared((BI,), "int32")
            m_ub = T.alloc_shared((VL,), accum_dtype)
            l_ub = T.alloc_shared((VL,), accum_dtype)
            alpha_ub = T.alloc_shared((VL,), accum_dtype)
            old_m_ub = T.alloc_shared((VL,), accum_dtype)
            O_ub = T.alloc_shared((ROWS, D), accum_dtype)
            O_tmp_ub = T.alloc_shared((ROWS, D), accum_dtype)
            O_out_ub = T.alloc_shared((ROWS, D), dtype)

            layouts = {P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)}
            if not cube_gather:
                layouts[KV_nz_ub] = make_ascend_compact_nz_layout(KV_nz_ub)
            T.annotate_layout(layouts)
            T.annotate_buffer_versions(
                {
                    KV_l1: versions["KV_l1"],
                    P_l1: versions["P_l1"],
                    qk_l0c: versions["qk_l0c"],
                    pv_l0c: versions["pv_l0c"],
                    S_ub: versions["S_ub"],
                    P_nz_ub: versions["P_nz_ub"],
                    val_ub: versions["val_ub"],
                    O_ub: versions["O_ub"],
                    O_tmp_ub: versions["O_tmp_ub"],
                }
            )
            # m/l are always touched now (always-merge reads them at
            # iteration 0 too); version 1 is fine for the in-order V-pipe
            # state chain at any NI.
            T.annotate_buffer_versions({m_ub: 1, l_ub: 1})
            if not cube_gather:
                T.annotate_buffer_versions(
                    {
                        KV_ub: versions["KV_ub"],
                        KV_nz_ub: versions["KV_nz_ub"],
                        idx_ub: versions["idx_ub"],
                    }
                )

            for task_block in T.Pipelined(
                NI,
                num_stages=stages,
                annotations={"enable_offset": True},
            ):
                b = task_block * BI
                if task_block == 0:
                    T.copy(Q[grp, g, :, :], Q_l1, l2_cache_ctrl=4)
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        zero = S.vdup(T.float32(0.0), "float32")
                        S.vsts(m_ub[0], S.vdup(T.float32(-(2.0**30)), "float32"), full, "NORM_B32")
                        S.vsts(l_ub[0], zero, full, "NORM_B32")
                        # Always-merge reads O_ub at iteration 0 (O =
                        # O*0 + tmp), so it must start at exact zero on
                        # both gather paths.
                        for r in range(ROWS):
                            for c in range(0, D, VL):
                                S.vsts(O_ub[r, c], zero, full, "NORM_B32")
                T.copy(Valid[grp, b : b + BI], val_ub)
                if cube_gather:
                    # Batch all 32 quad-base scalar reads up front (the
                    # assumes anchor the loads here so they emit as one
                    # run instead of interleaving with the DMAs), then
                    # share them between the K and V gathers. quad_ids
                    # iteration keeps this a trace-time Python loop.
                    bases = []
                    for q in quad_ids:
                        base = T.int32(Indices[grp, b + 4 * q])
                        T.assume(base >= 0)
                        T.assume(base < total_kv)
                        bases.append(base)
                    gather_kv(g, bases, K, KV_l1)
                else:
                    T.copy(Indices[grp, b : b + BI], idx_ub)
                    gather_kv_aiv(g, sid, K, idx_ub, KV_ub, KV_nz_ub, KV_l1)
                T.gemm(Q_l1, KV_l1, qk_l0c, transpose_B=True, clear_accum=True)
                T.dual_copy(qk_l0c, S_ub)
                if cube_gather:
                    gather_kv(g, bases, V, KV_l1)
                else:
                    gather_kv_aiv(g, sid, V, idx_ub, KV_ub, KV_nz_ub, KV_l1)
                softmax_pack(S_ub, val_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub)
                T.dual_copy(P_nz_ub[:ROWS, :], P_l1)
                T.gemm(P_l1, KV_l1, pv_l0c, transpose_B=False, clear_accum=True)
                # Always route pv through O_tmp and merge (both gather
                # paths). The task_block==0 direct-write branch makes the
                # O_tmp FIX->V edge irregular (present only for
                # task_block >= 1); InsertSync's flag placement is
                # schedule-shape-coupled and raced it at trip=2 AND at
                # M=16. A regular every-iteration edge gets the same
                # rotating flag ring as S_ub (proven at M=32-aiv, M=16-cube,
                # and in the generated C).
                T.dual_copy(pv_l0c, O_tmp_ub)
                with T.SimdVF():
                    full = S.pset(32, "PAT_ALL")
                    for r in range(ROWS):
                        a = S.vld(alpha_ub[r], "BRC_B32")
                        for c in range(0, D, VL):
                            o = S.vld(O_ub[r, c], "NORM")
                            t = S.vld(O_tmp_ub[r, c], "NORM")
                            S.vsts(
                                O_ub[r, c], S.vadd(S.vmul(o, a, full), t, full), full, "NORM_B32"
                            )
                if task_block == NI - 1:
                    with T.SimdVF():
                        # Guard l == 0 (fully invalid query) to 1 to avoid
                        # NaN, then cache 1/l.
                        full = S.pset(32, "PAT_ALL")
                        b16 = S.pset(16, "PAT_VL128")
                        lv = S.vld(l_ub[0], "NORM")
                        ones = S.vadds(S.vmuls(lv, T.float32(0.0), full), T.float32(1.0), full)
                        guard = S.vcmps(lv, T.float32(0.0), full, op="le")
                        denom = S.vsel(ones, lv, guard)
                        recip = S.vdiv(ones, denom, full)
                        S.vsts(l_ub[0], recip, full, "NORM_B32")
                        S.mem_bar("VST_VLD")
                        for r in range(ROWS):
                            rp = S.vld(l_ub[r], "BRC_B32")
                            for c in range(0, D, 2 * VL):
                                o0, o1 = S.vld2(O_ub[r, c], "DINTLV_B32")
                                f0 = S.vmul(o0, rp, full)
                                f1 = S.vmul(o1, rp, full)
                                c0 = S.vcvt(f0, dtype, full, sat=False, part=0)
                                c1 = S.vcvt(f1, dtype, full, sat=False, part=1)
                                S.vsts(
                                    O_out_ub[r, c],
                                    T.reinterpret(
                                        S.vor(
                                            T.reinterpret(c0, "uint16x128"),
                                            T.reinterpret(c1, "uint16x128"),
                                            b16,
                                        ),
                                        "bfloat16x128",
                                    ),
                                    b16,
                                    "NORM_B16",
                                )
                    # SimdVF blocks cannot touch GM; dual_copy stores AIV
                    # sid's half.
                    T.dual_copy(O_out_ub, O[grp, g, :, :])

    return main


# -- standalone correctness check and benchmark ---------------------------


def gen_indices(total_q, total_kv, top_k, fixed_budget=None, seed=0):
    """Random whole-block selections honoring the index contract."""
    torch.manual_seed(seed)
    num_blocks = total_kv // 4
    indices = torch.full((total_q, top_k), -1, dtype=torch.int32)
    if fixed_budget is not None:
        counts = torch.full((total_q,), fixed_budget, dtype=torch.int64)
    else:
        counts = torch.randint(1, top_k + 1, (total_q,))
    offsets = torch.arange(4, dtype=torch.int64)
    for t in range(total_q):
        n_blocks = min(int(counts[t]) // 4, num_blocks)
        blocks, _ = torch.randperm(num_blocks)[:n_blocks].sort()
        ids = blocks.unsqueeze(1) * 4 + offsets.unsqueeze(0)
        indices[t, : n_blocks * 4] = ids.reshape(-1).to(torch.int32)
    return indices


def split_indices(indices):
    """safe = clamped indices; valid = slot mask."""
    valid = (indices >= 0).to(torch.int32).contiguous()
    safe = indices.clamp_min(0).to(torch.int32).contiguous()
    return safe, valid


def pad_q_heads(q):
    """Regroup Q to (num_tokens, num_kv_heads, group, head_dim), zero-pad
    group to MP rows."""
    total_q, num_heads, head_dim = q.shape
    group = num_heads // NUM_KV_HEADS
    qp = q.reshape(total_q, NUM_KV_HEADS, group, head_dim)
    if group < PAD_M:
        pad = qp.new_zeros(total_q, NUM_KV_HEADS, PAD_M - group, head_dim)
        qp = torch.cat([qp, pad], dim=2)
    return qp.contiguous()


def ref_sparse_gqa(q, k, v, indices, sm_scale=None):
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group = num_heads // num_kv_heads
    scale = sm_scale if sm_scale is not None else head_dim**-0.5
    qf = q.float().view(total_q, num_kv_heads, group, head_dim)
    out = torch.zeros(total_q, num_heads, head_dim, dtype=torch.float32, device=q.device)
    for t in range(total_q):
        ids = indices[t][indices[t] >= 0].long()
        if ids.numel() == 0:
            continue
        kk = k[ids].float()
        vv = v[ids].float()
        s = torch.einsum("hgd,nhd->hgn", qf[t], kk) * scale
        p = s.softmax(dim=-1)
        out[t] = torch.einsum("hgn,nhd->hgd", p, vv).reshape(num_heads, head_dim)
    return out


def make_inputs(total_q, total_kv, top_k, seed=0, fixed_budget=None):
    torch.manual_seed(seed)
    q = torch.randn(total_q, NUM_HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    k = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    v = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    indices = gen_indices(total_q, total_kv, top_k, fixed_budget=fixed_budget, seed=seed).to("npu")
    return q, k, v, indices


def run(
    num_tokens,
    kv_length,
    top_k=2176,
    *,
    dynamic_shape=False,
    label=None,
    cube_gather=True,
    num_stages=None,
    chunk_rows=None,
):
    kernel = tilelang.compile(
        qsa_fwd(
            num_tokens,
            kv_length,
            top_k,
            dynamic_shape=dynamic_shape,
            cube_gather=cube_gather,
            num_stages=num_stages,
            chunk_rows=chunk_rows,
        ),
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    # Dump the generated C BEFORE verification so a failed assert still
    # leaves the artifact for sync-pairing audits (out.md turned out to be
    # the stale first-iteration compile-failure dump, not this kernel).
    tag = label or f"q{num_tokens}-kv{kv_length}"
    with open(f"kernel_{tag}.c", "w", encoding="utf-8") as f:
        f.write(kernel.get_kernel_source())
    q, k, v, indices = make_inputs(num_tokens, kv_length, top_k, seed=0)
    # Contract corners: fully invalid query, short leading run, tail quad
    # ending at the KV boundary.
    if num_tokens > 0:
        indices[0] = -1
    if num_tokens > 1:
        indices[1, :8] = -1
    if num_tokens > 2:
        indices[2, top_k - 4 :] = torch.tensor(
            [kv_length - 3, kv_length - 2, kv_length - 1, -1], dtype=torch.int32
        )
    expected = ref_sparse_gqa(q, k, v, indices)
    safe, valid = split_indices(indices)
    # The cube gather relies on KV_PAD_ROWS zero rows (see gather_kv); the
    # AIV path never reads them. Pad AFTER the reference computation.
    kp = torch.cat([k, k.new_zeros(KV_PAD_ROWS, *k.shape[1:])])
    vp = torch.cat([v, v.new_zeros(KV_PAD_ROWS, *v.shape[1:])])
    actual = kernel(pad_q_heads(q), kp, vp, safe, valid)
    group = NUM_HEADS // NUM_KV_HEADS
    actual = actual[:, :, :group, :].reshape(num_tokens, NUM_HEADS, HEAD_DIM)
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)
    print(
        f"Verification passed; max error: " f"{(actual.float() - expected).abs().max().item():.6f}"
    )

    q, k, v, indices = make_inputs(
        num_tokens, kv_length, top_k, seed=1, fixed_budget=min(2051, top_k)
    )
    safe, valid = split_indices(indices)
    qp = pad_q_heads(q)
    kp = torch.cat([k, k.new_zeros(KV_PAD_ROWS, *k.shape[1:])])
    vp = torch.cat([v, v.new_zeros(KV_PAD_ROWS, *v.shape[1:])])
    kernel(qp, kp, vp, safe, valid)
    latency_ms = do_bench(
        lambda: kernel(qp, kp, vp, safe, valid), backend="msprof", _n_warmup=30, _n_repeat=20
    )
    # Useful flops: 24 real query heads x average valid slots per query
    # (QSA selects fewer than top_k slots; pad rows do no useful work).
    n_valid = float((indices >= 0).sum()) / num_tokens
    flops = 4 * num_tokens * NUM_HEADS * HEAD_DIM * n_valid
    label = label or f"q{num_tokens}-kv{kv_length}"
    print(
        f"[{label}] {latency_ms * 1000:.2f} us/iter | "
        f"{flops / (latency_ms * 1e-3) / 1e12:.1f} TFLOPS"
    )


def main():
    global PAD_M
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-tokens", type=int, help="run a single custom shape instead of the default scenarios"
    )
    parser.add_argument(
        "--kv-length", type=int, help="KV length for the custom shape (required with --num-tokens)"
    )
    parser.add_argument("--top-k", type=int, default=2176)
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="symbolic num_tokens/kv_length (one kernel serves all shapes)",
    )
    parser.add_argument(
        "--aiv-gather",
        action="store_true",
        help="use the legacy AIV UB-relay gather (A/B experiment) " "instead of Cube-side GM->L1",
    )
    parser.add_argument(
        "--stages",
        type=int,
        help="override the software-pipeline depth "
        "(e.g. 3 to launch the gather two iterations ahead; "
        "KV_l1 already carries 3 versions)",
    )
    parser.add_argument(
        "--pad-m",
        type=int,
        choices=(16, 32),
        default=PAD_M,
        help="gemm M rows (padded group). 32 = baseline; "
        "16 = GBSA-parity experiment (halves all gemm/vector "
        "work for the 12-head group)",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        choices=(32, 64),
        help="AIV-relay staging chunk (aiv path, NI > 2 only). "
        "64 halves the nz-scatter/ub2l1 count; needs the "
        "UB headroom that --pad-m 16 frees",
    )
    args = parser.parse_args()
    cube = not args.aiv_gather
    PAD_M = args.pad_m

    if args.num_tokens is not None:
        assert args.kv_length is not None, "--kv-length is required with --num-tokens"
        run(
            args.num_tokens,
            args.kv_length,
            args.top_k,
            dynamic_shape=args.dynamic,
            cube_gather=cube,
            num_stages=args.stages,
            chunk_rows=args.chunk_rows,
        )
        return
    # The small run covers the sequential path (NI <= 2); decode/prefill
    # cover the pipelined path (NI = 17) and the two real-world scenarios.
    run(
        16,
        2048,
        256,
        dynamic_shape=args.dynamic,
        label="correctness-small",
        cube_gather=cube,
        num_stages=args.stages,
        chunk_rows=args.chunk_rows,
    )
    run(
        128,
        65536,
        args.top_k,
        dynamic_shape=args.dynamic,
        label="decode-qwen3.8",
        cube_gather=cube,
        num_stages=args.stages,
        chunk_rows=args.chunk_rows,
    )
    run(
        4096,
        32768,
        args.top_k,
        dynamic_shape=args.dynamic,
        label="prefill-qwen3.8",
        cube_gather=cube,
        num_stages=args.stages,
        chunk_rows=args.chunk_rows,
    )


if __name__ == "__main__":
    main()
