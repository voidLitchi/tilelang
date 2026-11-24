"""Persistent mixed-core QSA schedule for Qwen3.8-Flash-Next on Ascend 950."""

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tvm import tirx


NUM_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
GROUP_SIZE = NUM_HEADS // NUM_KV_HEADS
LOGICAL_TOP_K = 2051
PHYSICAL_TOP_K = 2176
PAD_M = 16
PIPELINE_STAGES = 3
SUPPORTED_AIC_COUNTS = (28, 32, 36)

# Offsets are elements of the corresponding typed view. Both AIVs have the
# same private UB map. The AIC's FIX transfers address those maps by M-split.
NZ, ND, P_PACK, OUT = 0, 16896, 49664, 51968  # BF16
PV, SCORES, ACC, MAX, SUM, ALPHA, OLD_MAX = 16640, 20736, 22784, 27520, 27584, 27648, 27712  # FP32
INDEX, VALID = 27008, 27264  # int32
UB_BYTES = (OLD_MAX + 64) * 4
Q_L1, P_L1 = 98304, 102400  # BF16, after three [128, 256] K/V slots


def _op(name, *args):
    """Emit a registered Ascend tile intrinsic with explicit instruction operands."""
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_" + name), *args)


def _load_rows(dst, src, rows=1, src_stride=512):
    return _op("copy_gm_to_ubuf", T.access_ptr(dst, "w", extent=rows * 256), src, 0, rows, 512, 0, 0, 0, 0, src_stride, 512)


def _fix_split(dst, src, width):
    # The FIX engine writes eight rows to each AIV. No quantization or unit flag.
    return _op(
        "copy_matrix_cc_to_ub",
        T.access_ptr(dst, "w", extent=8 * width),
        T.access_ptr(src, "r", extent=16 * width),
        0,
        width,
        16,
        width,
        16,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        *([0] * 11),
    )


def _arrive(pipe, event):
    return T.ascend_cross_core_set_flag(4, pipe, event)


def _await_peer(pipe, event):
    return T.ascend_cross_core_wait_flag(4, pipe, event)


@T.macro
def _initialize(f):
    with T.SimdVF():
        full = S.pset(32)
        zero = S.vdup(0.0, "float32", full)
        S.vsts(f[MAX], S.vdup(-1073741824.0, "float32", full), full)
        S.vsts(f[SUM], zero, full)
        for row in range(8):
            for col in range(4):
                S.vsts(f[ACC + row * 256 + col * 64], zero, full)


@T.macro
def _softmax(f, ints, bf, parity):
    with T.SimdVF():
        full = S.pset(32)
        one = S.pset(32, "PAT_VL1")
        b16 = S.pset(16, "PAT_VL128")
        valid0 = S.vcvt(S.vld(ints[VALID + parity * 128]), "float32", full)
        valid1 = S.vcvt(S.vld(ints[VALID + parity * 128 + 64]), "float32", full)
        pen0 = S.vmuls(S.vadds(valid0, -1.0, full), 1e30, full)
        pen1 = S.vmuls(S.vadds(valid1, -1.0, full), 1e30, full)
        dst = S.make_ubuf_ptr(T.access_ptr(bf[P_PACK + parity * 1152], "w", extent=1152), "bfloat16")
        for row in range(8):
            offset = SCORES + parity * 1024 + row * 128
            s0 = S.vadd(S.vmuls(S.vld(f[offset]), 0.0625, full), pen0, full)
            s1 = S.vadd(S.vmuls(S.vld(f[offset + 64]), 0.0625, full), pen1, full)
            S.vsts(f[offset], s0, full)
            S.vsts(f[offset + 64], s1, full)
            old = S.vld(f[MAX + row], "BRC_B32")
            S.vsts(f[OLD_MAX + row], old, one, "ONEPT_B32")
            m = S.vcmax(S.vmax(S.vmax(s0, s1, full), old, full), full)
            S.vsts(f[MAX + row], m, one, "ONEPT_B32")
        S.mem_bar("VST_VLD")
        for row in range(8):
            m = S.vld(f[MAX + row], "BRC_B32")
            even, odd = S.vld2(f[SCORES + parity * 1024 + row * 128], "DINTLV_B32")
            e0 = S.vexpdif(even, m, full)
            e1 = S.vexpdif(odd, m, full)
            pair_sum = S.vadd(e0, e1, full)
            b0 = S.vcvt(e0, "bfloat16", full, sat=False, part=0)
            b1 = S.vcvt(e1, "bfloat16", full, sat=False, part=1)
            bits = S.vor(T.reinterpret(b0, "uint16x128"), T.reinterpret(b1, "uint16x128"), b16)
            dst = S.vsstb(T.reinterpret(bits, "bfloat16x128"), dst, (9 << 16) | 1, b16, update=True)
            psum = S.vcadd(pair_sum, full)
            alpha = S.vexpdif(S.vld(f[OLD_MAX + row], "BRC_B32"), m, full)
            S.vsts(f[ALPHA + row], alpha, one, "ONEPT_B32")
            new_sum = S.vadd(S.vmul(S.vld(f[SUM + row], "BRC_B32"), alpha, full), psum, one)
            S.vsts(f[SUM + row], new_sum, one, "ONEPT_B32")


@T.macro
def _accumulate(f, parity):
    with T.SimdVF():
        full = S.pset(32)
        for row in range(8):
            alpha = S.vld(f[ALPHA + row], "BRC_B32")
            for col in range(4):
                idx = row * 256 + col * 64
                prev = S.vld(f[ACC + idx])
                cur = S.vld(f[PV + parity * 2048 + idx])
                S.vsts(f[ACC + idx], S.vadd(S.vmul(prev, alpha, full), cur, full), full)


@T.macro
def _normalize(f, bf):
    with T.SimdVF():
        full = S.pset(32)
        b16 = S.pset(16, "PAT_VL128")
        sums = S.vld(f[SUM])
        ones = S.vadds(S.vmuls(sums, 0.0, full), 1.0, full)
        denom = S.vsel(ones, sums, S.vcmps(sums, 0.0, full, "le"))
        S.vsts(f[SUM], S.vdiv(ones, denom, full), full)
        S.mem_bar("VST_VLD")
        for row in range(8):
            recip = S.vld(f[SUM + row], "BRC_B32")
            for col in range(2):
                idx = row * 256 + col * 128
                even, odd = S.vld2(f[ACC + idx], "DINTLV_B32")
                b0 = S.vcvt(S.vmul(even, recip, full), "bfloat16", full, sat=False, part=0)
                b1 = S.vcvt(S.vmul(odd, recip, full), "bfloat16", full, sat=False, part=1)
                bits = S.vor(T.reinterpret(b0, "uint16x128"), T.reinterpret(b1, "uint16x128"), b16)
                S.vsts(bf[OUT + idx], T.reinterpret(bits, "bfloat16x128"), b16, "NORM_B16")


@T.macro
def _cube(core, cores, tiles, q, l1, a, b, c, f):
    T.ascend_set_flag("M_MTE1", 0)
    T.ascend_set_flag("M_MTE1", 1)
    T.ascend_set_flag("MTE1_MTE2", 0)
    # One PV is issued one tick behind its QK. The last tick only drains PV.
    for tick in range(tiles + 1):
        next_slot = tick & 1
        prev_slot = (tick + 1) & 1
        if tick < tiles:
            T.ascend_wait_flag("MTE1_MTE2", 0)
            if tick % 17 == 0:
                task = core + tick // 17 * cores
                _op(
                    "copy_gm_to_cbuf",
                    T.access_ptr(l1[Q_L1], "w", extent=4096),
                    q.access_ptr("r", offset=task * 4096, extent=4096),
                    0,
                    512,
                    4,
                    16,
                    256,
                    0,
                    0,
                    0,
                    16,
                    "",
                )
            T.ascend_set_flag("MTE2_MTE1", 0)
            for part in range(2):
                if part == 0:
                    _await_peer("PIPE_MTE1", 22 + next_slot)
                    _await_peer("PIPE_MTE1", 6 + next_slot)
                T.ascend_wait_flag("M_MTE1", 0)
                _op(
                    "load_cbuf_to_cb",
                    T.access_ptr(b[0], "w", extent=16384),
                    T.access_ptr(l1[(tick % 3) * 32768], "r", extent=32768),
                    0,
                    part * 8,
                    8,
                    8,
                    8,
                    8,
                    0,
                )
                if part == 1:
                    _arrive("PIPE_MTE1", 4 + next_slot)
                    _arrive("PIPE_MTE1", 20 + next_slot)
                if part == 0:
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                _op(
                    "load_cbuf_to_ca",
                    T.access_ptr(a[0], "w", extent=2048),
                    T.access_ptr(l1[Q_L1], "r", extent=4096),
                    0,
                    part * 8,
                    1,
                    8,
                    1,
                    1,
                    0,
                )
                T.ascend_set_flag("MTE1_M", 0)
                if part == 1:
                    T.ascend_set_flag("MTE1_MTE2", 0)
                T.ascend_wait_flag("MTE1_M", 0)
                _op(
                    "mad",
                    T.access_ptr(c[8192 + next_slot * 2048], "rw", extent=2048),
                    T.access_ptr(a[0], "r", extent=2048),
                    T.access_ptr(b[0], "r", extent=16384),
                    16,
                    128,
                    128,
                    0,
                    1,
                    0,
                    part == 0,
                )
                T.ascend_set_flag("M_MTE1", 0)
                if part == 1:
                    T.ascend_set_flag("M_FIX", next_slot)
        if tick >= 1:
            for part in range(2):
                if part == 0:
                    _await_peer("PIPE_MTE1", 18 + prev_slot)
                    _await_peer("PIPE_MTE1", 2 + prev_slot)
                T.ascend_wait_flag("M_MTE1", 1)
                _op(
                    "load_cbuf_to_cb",
                    T.access_ptr(b[16384], "w", extent=16384),
                    T.access_ptr(l1[((tick + 2) % 3) * 32768], "r", extent=32768),
                    part * 4,
                    0,
                    4,
                    16,
                    8,
                    16,
                    1,
                )
                if part == 0:
                    _await_peer("PIPE_MTE1", 25)
                    _await_peer("PIPE_MTE1", 9)
                _op(
                    "load_cbuf_to_ca",
                    T.access_ptr(a[2048], "w", extent=1024),
                    T.access_ptr(l1[P_L1 + prev_slot * 2048], "r", extent=2048),
                    0,
                    part * 4,
                    1,
                    4,
                    1,
                    1,
                    0,
                )
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 1)
                _op(
                    "mad",
                    T.access_ptr(c[prev_slot * 4096], "rw", extent=4096),
                    T.access_ptr(a[2048], "r", extent=1024),
                    T.access_ptr(b[16384], "r", extent=16384),
                    16,
                    64,
                    256,
                    0,
                    1,
                    0,
                    part == 0,
                )
                T.ascend_set_flag("M_MTE1", 1)
                if part == 1:
                    T.ascend_set_flag("M_FIX", 2)
            T.ascend_wait_flag("M_FIX", 2)
            _fix_split(f[PV + prev_slot * 2048], c[prev_slot * 4096], 256)
            _arrive("PIPE_FIX", 8)
            _arrive("PIPE_FIX", 24)
        if tick < tiles:
            T.ascend_wait_flag("M_FIX", next_slot)
            _fix_split(f[SCORES + next_slot * 1024], c[8192 + next_slot * 2048], 128)
            _arrive("PIPE_FIX", next_slot)
            _arrive("PIPE_FIX", next_slot + 16)
    T.ascend_wait_flag("M_MTE1", 0)
    T.ascend_wait_flag("M_MTE1", 1)
    T.ascend_wait_flag("MTE1_MTE2", 0)


@T.macro
def _relay(is_key, tile, core, sid, last_quad, kv, ints, bf, l1):
    parity = tile & 1
    for chunk in range(2):
        # Each relay issues exactly two chunks, so its ND and NZ slots always
        # start at zero and advance together. The 33rd NZ row is bank padding.
        tail = tile % 17 == 16 and chunk == 1
        for quad in range(8):
            if is_key:
                if chunk == 0 and quad == 0:
                    T.ascend_wait_flag("MTE2_S", parity)
                if tail and quad == 0:
                    T.ascend_wait_flag("MTE2_S", parity + 2)
            offset = parity * 128 + sid * 64 + chunk * 32 + quad * 4
            base = ints[INDEX + offset]
            all_valid = T.alloc_var("int32", init=1)
            if tail:
                for row in range(4):
                    all_valid = all_valid & ints[VALID + offset + row]
            if quad == 0:
                T.ascend_wait_flag("V_MTE2", chunk)
            dst = ND + chunk * 8192 + quad * 1024
            if base <= last_quad and all_valid != 0:
                _load_rows(bf[dst], kv.access_ptr("r", offset=base * 512 + (core & 1) * 256, extent=1792), 4, 1024)
            else:
                for row in range(4):
                    if not tail or ints[VALID + offset + row] != 0:
                        idx = ints[INDEX + offset + row]
                        _load_rows(bf[dst + row * 256], kv.access_ptr("r", offset=idx * 512 + (core & 1) * 256, extent=256))
            if quad == 7:
                T.ascend_set_flag("MTE2_V", chunk)
                if not is_key and chunk == 1:
                    T.ascend_set_flag("S_MTE2", parity)
        T.ascend_wait_flag("MTE3_V", chunk)
        T.ascend_wait_flag("MTE2_V", chunk)
        _op(
            "nd2nz_scatter",
            T.access_ptr(bf[ND + chunk * 8192], "r", extent=8192),
            T.access_ptr(bf[NZ + chunk * 8448], "w", extent=8448),
            32,
            256,
            33,
            "bfloat16_t",
            "bfloat16_t",
        )
        T.ascend_set_flag("V_MTE3", chunk)
        T.ascend_set_flag("V_MTE2", chunk)
        if not is_key and chunk == 0:
            _await_peer("PIPE_MTE3", parity + 4)
        T.ascend_wait_flag("V_MTE3", chunk)
        T.ascend_nd2nz_post_copy(
            l1[(tile % 3) * 32768 + sid * 1024 + chunk * 512], bf[NZ + chunk * 8448], 32, 256, src_pitch=33, dst_pitch=128, dtype="bfloat16"
        )
        T.ascend_set_flag("MTE3_V", chunk)
        if chunk == 1:
            _arrive("PIPE_MTE3", parity + (6 if is_key else 2))


@T.macro
def _vector(core, sid, cores, tiles, last_quad, indices, valid, key, value, out, bf, ints, f, l1):
    for slot in (0, 1):
        T.ascend_set_flag("MTE3_V", slot)
    for slot in (0, 1, 2, 3):
        T.ascend_set_flag("V_MTE2", slot)
    for slot in (0, 1):
        T.ascend_set_flag("S_MTE2", slot)
    for tick in range(tiles + 5):
        if tick >= 5:
            T.ascend_wait_flag("V_MTE3", 2)
            if (tick - 5) % 17 == 16:
                task = core + (tick - 5) // 17 * cores
                _op(
                    "copy_ubuf_to_gm",
                    out.access_ptr("w", offset=task * 4096 + sid * 2048, extent=2048),
                    T.access_ptr(bf[OUT], "r", extent=2048),
                    0,
                    1,
                    4096,
                    0,
                    4096,
                    4096,
                )
                T.ascend_set_flag("MTE3_V", 4)
        if tick >= 4 and tick < tiles + 4:
            if (tick - 4) % 17 == 16:
                if tick > 20:
                    T.ascend_wait_flag("MTE3_V", 4)
                _normalize(f, bf)
            T.ascend_set_flag("V_MTE3", 2)
        if tick >= 3 and tick < tiles + 3:
            if (tick - 3) % 17 == 0:
                _initialize(f)
            parity = (tick + 1) & 1
            _await_peer("PIPE_V", parity)
            _softmax(f, ints, bf, parity)
            T.ascend_set_flag("V_MTE3", 3)
            T.ascend_set_flag("V_MTE2", parity + 2)
            T.ascend_wait_flag("V_MTE3", 3)
            T.ascend_nd2nz_post_copy(
                l1[P_L1 + parity * 2048 + sid * 128], bf[P_PACK + parity * 1152], 8, 128, src_pitch=9, dst_pitch=16, dtype="bfloat16"
            )
            _arrive("PIPE_MTE3", 9)
        if tick >= 2 and tick < tiles + 2:
            _relay(False, tick - 2, core, sid, last_quad, value, ints, bf, l1)
        if tick >= 1 and tick < tiles + 1:
            tile = tick - 1
            parity = tile & 1
            query = (core + tile // 17 * cores) // 2
            T.ascend_wait_flag("V_MTE2", parity + 2)
            T.copy(valid[query, (tile % 17) * 128 : (tile % 17 + 1) * 128], ints[VALID + parity * 128 : VALID + (parity + 1) * 128])
            if tile % 17 == 16:
                T.ascend_set_flag("MTE2_S", parity + 2)
            _relay(True, tile, core, sid, last_quad, key, ints, bf, l1)
        if tick >= 3 and tick < tiles + 3:
            _await_peer("PIPE_V", 8)
            _accumulate(f, (tick + 1) & 1)
        if tick < tiles:
            parity = tick & 1
            query = (core + tick // 17 * cores) // 2
            T.ascend_wait_flag("S_MTE2", parity)
            T.copy(indices[query, (tick % 17) * 128 : (tick % 17 + 1) * 128], ints[INDEX + parity * 128 : INDEX + (parity + 1) * 128])
            T.ascend_set_flag("MTE2_S", parity)
    for slot in (0, 1):
        T.ascend_wait_flag("MTE3_V", slot)
    for slot in (0, 1, 2, 3):
        T.ascend_wait_flag("V_MTE2", slot)
    T.ascend_wait_flag("MTE3_V", 4)
    for slot in (0, 1):
        T.ascend_wait_flag("S_MTE2", slot)


def qsa_fwd(total_q, total_kv, *, aic_count=32, dynamic_shape=False):
    assert aic_count in SUPPORTED_AIC_COUNTS, "aic_count must be 28, 32, or 36"
    assert total_q > 0 and total_kv >= 4, "Q must be positive and KV must be at least four rows"
    if dynamic_shape:
        total_q = T.dynamic("num_tokens")
        padded_kv = T.dynamic("padded_kv_length")
    else:
        padded_kv = total_kv + 3
        assert max(total_q * 8192, padded_kv * 512) <= 2**31, "GM offsets must fit int32"

    @T.prim_func
    def main(
        Q: T.Buffer((total_q, NUM_KV_HEADS, PAD_M, HEAD_DIM), "bfloat16"),
        K: T.Buffer((padded_kv, NUM_KV_HEADS, HEAD_DIM), "bfloat16"),
        V: T.Buffer((padded_kv, NUM_KV_HEADS, HEAD_DIM), "bfloat16"),
        Indices: T.Buffer((total_q, PHYSICAL_TOP_K), "int32"),
        Valid: T.Buffer((total_q, PHYSICAL_TOP_K), "int32"),
        O: T.Buffer((total_q, NUM_KV_HEADS, PAD_M, HEAD_DIM), "bfloat16"),
    ):
        with T.Kernel(aic_count) as core:
            ub = T.alloc_shared((UB_BYTES,), "uint8")
            bf = T.view(ub, (UB_BYTES // 2,), "bfloat16")
            ints = T.view(ub, (UB_BYTES // 4,), "int32")
            f = T.view(ub, (UB_BYTES // 4,), "float32")
            l1 = T.alloc_l1((P_L1 + 4096,), "bfloat16")
            a = T.alloc_l0a((3072,), "bfloat16")
            b = T.alloc_l0b((32768,), "bfloat16")
            c = T.alloc_l0c((12288,), "float32", layout=False)
            tasks = total_q * 2
            tiles = 17 * (T.truncdiv(tasks - 1 - core, aic_count) + 1)
            with T.Cube():
                if core < tasks:
                    _cube(core, aic_count, tiles, Q, l1, a, b, c, f)
            with T.Vector() as sid:
                if core < tasks:
                    _vector(core, sid, aic_count, tiles, padded_kv - 7, Indices, Valid, K, V, O, bf, ints, f, l1)

    return main
