"""Verification, CCE audit, and benchmark script for qsmall_varlen_attn_fwd (Ascend NPU).

Covers all functional, dynamic-shape, boundary, int64-overflow, and performance
requirements from `qsmall_varlen_attn_fwd_spec.md` (T1-T5 + benchmark).

Usage:
    # 1) Offline compile + CCE audit + CPU Golden self-check (no NPU required, e.g. local WSL):
    python verify_and_bench.py --compile-only

    # 2) Full on-device verification (T1-T5) + baseline benchmark on Ascend 950PR (28AIC/56AIV)
    #    or Ascend 950DT (32AIC/64AIV):
    python verify_and_bench.py --bench

    # 3) Explicitly pin AIV core count or sweep tuning configs on NPU:
    python verify_and_bench.py --num-aiv 56 --bench
    python verify_and_bench.py --num-aiv 64 --bench --sweep

    # 4) Lightweight baseline-only run for on-device msprof / msprof op profiling:
    msprof --output=./prof_qsmall --aic-metrics=PipeUtilization --sys-hardware-mem=on \
      --application="python verify_and_bench.py --profile-only --num-aiv 64"
"""

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

import torch

# Allow direct execution without relying on top-level `examples` package shadowing.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from kernel import (  # noqa: E402
    DEFAULT_BLOCK_N,
    DEFAULT_L2_CACHE_CTRL,
    DEFAULT_NUM_STAGES,
    HEAD_DIM,
    NUM_HEADS,
    clear_kernel_cache,
    detect_num_aiv,
    get_compile_count,
    get_compiled_kernel,
    qsmall_varlen_attn_fwd,
)

BASELINE_N = 1280
BASELINE_SEQLEN = 578
BASELINE_TOTAL_K = BASELINE_N * BASELINE_SEQLEN  # 739,840
BASELINE_SEED = 7
BASELINE_TOTAL_BYTES_GB = 5.313  # Spec §4.1 decimal GB (1 GB = 1e9 B)


# ---------------------------------------------------------------------------
# CPU FP32 Golden Oracle (Spec §7.1)
# ---------------------------------------------------------------------------
def golden_qsmall_varlen_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact CPU FP32 reference matching Spec §3.1, §3.2, and §7.1.

    Inputs are expected to be already quantized to bfloat16 before calling.
    """
    q_f32 = q.detach().cpu().float()
    k_f32 = k.detach().cpu().float()
    v_f32 = v.detach().cpu().float()
    cu = cu_seqlens_k.detach().cpu().to(torch.int64).tolist()

    n_samples, h, d = q_f32.shape
    out_ref = torch.zeros((n_samples, h, d), dtype=torch.bfloat16)
    lse_ref = torch.full((n_samples, h), float("-inf"), dtype=torch.float32)

    for i in range(n_samples):
        k_start = cu[i]
        k_end = cu[i + 1]
        if k_end <= k_start:
            continue
        qi = q_f32[i]  # [H, D]
        ki = k_f32[k_start:k_end]  # [S, H, D]
        vi = v_f32[k_start:k_end]  # [S, H, D]
        # scores: [H, S]
        scores = torch.einsum("hd,shd->hs", qi, ki) * float(sm_scale)
        lse_i = torch.logsumexp(scores, dim=-1)  # [H], natural log
        probs = torch.softmax(scores, dim=-1)  # [H, S]
        oi = torch.einsum("hs,shd->hd", probs, vi)  # [H, D]
        out_ref[i] = oi.to(torch.bfloat16)
        lse_ref[i] = lse_i

    return out_ref, lse_ref


def make_inputs(
    seqlens: list[int],
    *,
    seed: int = BASELINE_SEED,
    prefix_pad: int = 0,
    suffix_pad: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate quantized bfloat16 Q/K/V and int32 cu_seqlens_k deterministically."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_samples = len(seqlens)
    cu_list = [prefix_pad]
    for s in seqlens:
        cu_list.append(cu_list[-1] + int(s))
    total_k = cu_list[-1] + suffix_pad

    # Quantize on CPU first so Golden and Device observe identical BF16 bit patterns (Spec §7.1).
    q = torch.randn((n_samples, NUM_HEADS, HEAD_DIM), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn((total_k, NUM_HEADS, HEAD_DIM), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn((total_k, NUM_HEADS, HEAD_DIM), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    cu_seqlens_k = torch.tensor(cu_list, dtype=torch.int32)

    if str(device) != "cpu":
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        cu_seqlens_k = cu_seqlens_k.to(device)
    return q, k, v, cu_seqlens_k


def check_accuracy(
    name: str,
    dev_out: torch.Tensor,
    dev_lse: torch.Tensor,
    ref_out: torch.Tensor,
    ref_lse: torch.Tensor,
    seqlens: list[int],
) -> None:
    """Verify Spec §7.2 accuracy thresholds for both out (bf16) and lse (fp32)."""
    out_dev_f = dev_out.detach().cpu().float()
    out_ref_f = ref_out.detach().cpu().float()
    lse_dev_f = dev_lse.detach().cpu().float()
    lse_ref_f = ref_lse.detach().cpu().float()

    assert not torch.isnan(out_dev_f).any(), f"[{name}] NaN detected in out!"
    assert not torch.isnan(lse_dev_f).any(), f"[{name}] NaN detected in lse!"

    # Check empty sequences strictly (out == 0.0, lse == -inf)
    for i, s in enumerate(seqlens):
        if s == 0:
            assert torch.equal(out_dev_f[i], torch.zeros_like(out_dev_f[i])), (
                f"[{name}] Empty sequence sample {i}: out must be strictly 0.0, max abs={out_dev_f[i].abs().max().item()}"
            )
            assert torch.isneginf(lse_dev_f[i]).all(), (
                f"[{name}] Empty sequence sample {i}: lse must be strictly -inf, got {lse_dev_f[i][:4]}"
            )

    out_diff = (out_dev_f - out_ref_f).abs()
    out_max_abs = out_diff.max().item()
    out_rel_l2 = (out_diff.pow(2).sum().sqrt() / (out_ref_f.pow(2).sum().sqrt() + 1e-12)).item()
    out_ok = torch.allclose(out_dev_f, out_ref_f, rtol=1e-2, atol=1e-2)

    finite_mask = torch.isfinite(lse_ref_f)
    if finite_mask.any():
        lse_diff = (lse_dev_f[finite_mask] - lse_ref_f[finite_mask]).abs()
        lse_max_abs = lse_diff.max().item()
        lse_ok = torch.allclose(lse_dev_f[finite_mask], lse_ref_f[finite_mask], rtol=1e-3, atol=1e-3)
    else:
        lse_max_abs = 0.0
        lse_ok = True

    neg_inf_mask = torch.isneginf(lse_ref_f)
    if neg_inf_mask.any():
        lse_ok = lse_ok and bool(torch.isneginf(lse_dev_f[neg_inf_mask]).all().item())

    status = "PASS" if (out_ok and lse_ok) else "FAIL"
    print(
        f"[{status}] {name}: "
        f"out(max_abs={out_max_abs:.4e}, rel_l2={out_rel_l2:.4e}) | "
        f"lse(max_abs={lse_max_abs:.4e})"
    )
    if not (out_ok and lse_ok):
        raise AssertionError(f"[{name}] Accuracy check failed! out_ok={out_ok}, lse_ok={lse_ok}")


# ---------------------------------------------------------------------------
# Offline CCE Static Audit (runs without NPU)
# ---------------------------------------------------------------------------
def audit_compiled_cce(
    block_n: int,
    num_stages: int,
    l2_cache_ctrl: str,
    *,
    dump_path: str | None = None,
) -> str:
    """Compile kernel and statically audit generated Ascend CCE source."""
    clear_kernel_cache()
    t0 = time.perf_counter()
    kernel = get_compiled_kernel(
        block_n=block_n,
        num_stages=num_stages,
        l2_cache_ctrl=l2_cache_ctrl,
    )
    dt = time.perf_counter() - t0
    src = kernel.get_kernel_source()

    if dump_path:
        Path(dump_path).write_text(src, encoding="utf-8")

    # 1. Pure __vector__ kernel with dynamic N, total_k (int64_t), sm_scale (float), grid_n (int32_t)
    sig_match = re.search(r"extern \"C\" __global__ __vector__ void \w+\(([^)]+)\)", src)
    assert sig_match is not None, "Expected pure __vector__ global kernel signature in CCE!"
    sig = sig_match.group(1)
    assert "int64_t total_k" in sig, f"Expected int64_t total_k in signature, got: {sig}"
    assert "int32_t N" in sig, f"Expected int32_t N in signature, got: {sig}"
    assert "int32_t grid_n" in sig, f"Expected int32_t grid_n in signature, got: {sig}"
    assert "float sm_scale" in sig, f"Expected float sm_scale in signature, got: {sig}"

    # 2. Check GM->UB DMA uses 512B burst & 3584B row stride for K/V
    kv_copies = re.findall(r"asc_copy_gm2ub_align\([^\n]+3584,\s*512\);", src)
    assert len(kv_copies) >= 2, (
        f"Expected >= 2 strided K/V GM->UB copies with 512B burst & 3584B stride, found {len(kv_copies)}"
    )

    # 3. Check K/V pointer indexing uses int64_t arithmetic (prevent int32 overflow for T5)
    for line in kv_copies:
        assert "(int64_t)1792" in line, f"Missing int64_t row stride (1792) in K/V copy: {line}"

    # 4. Check key SIMD instructions were emitted
    for inst in ("simd_inst::vcvt", "simd_inst::vcgadd", "simd_inst::vcpadd", "simd_inst::vexpdif", "simd_inst::vmula", "simd_inst::vselr"):
        assert inst in src, f"Expected SIMD instruction `{inst}` in generated CCE!"

    # 5. Estimate static UB footprint from highest buf_dyn_shmem byte offset
    byte_offsets: list[int] = []
    for elem_type, elem_size in (("bfloat16_t", 2), ("float", 4), ("uint8_t", 1), ("int32_t", 4)):
        for m in re.finditer(rf"\(\(__ubuf__ {elem_type}\*\)buf_dyn_shmem\)\[(\d+)\]", src):
            byte_offsets.append(int(m.group(1)) * elem_size)
    max_ub_offset = max(byte_offsets) if byte_offsets else 0
    ub_limit = 253952  # 248 KiB (Ascend 950 VectorCoreSpec.ub_size)

    # 6. Check multi-buffer sync flags exist when num_stages >= 2
    has_pipe_sync = "asc_sync_wait(PIPE_V, PIPE_MTE2" in src and "asc_sync_notify(PIPE_MTE2, PIPE_V" in src
    assert has_pipe_sync, "Expected PIPE_MTE2 <-> PIPE_V synchronization flags in CCE!"

    print(
        f"[CCE AUDIT PASS] block_n={block_n}, num_stages={num_stages}, l2={l2_cache_ctrl} | "
        f"compile={dt:.2f}s | src={len(src)} chars | kv_dma_sites={len(kv_copies)} | "
        f"max_static_ub_offset={max_ub_offset / 1024:.1f} KiB / {ub_limit / 1024:.0f} KiB"
    )
    assert max_ub_offset < ub_limit - 8192, (
        f"Static UB offset {max_ub_offset} B is too close to 248 KiB limit ({ub_limit} B)!"
    )
    return src


def run_cpu_golden_self_check() -> None:
    """Validate CPU Golden Oracle on edge cases (empty seq, tail, uncovered buffer)."""
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    seqlens = [0, 1, 2, 15, 32, 64, 65, 130]
    q, k, v, cu = make_inputs(seqlens, seed=123, prefix_pad=17, suffix_pad=29, device="cpu")
    out, lse = golden_qsmall_varlen_attn_fwd(q, k, v, cu, sm_scale)
    assert out.shape == (len(seqlens), NUM_HEADS, HEAD_DIM) and out.dtype == torch.bfloat16
    assert lse.shape == (len(seqlens), NUM_HEADS) and lse.dtype == torch.float32
    assert torch.equal(out[0], torch.zeros_like(out[0]))
    assert torch.isneginf(lse[0]).all()
    assert torch.isfinite(out[1:].float()).all() and torch.isfinite(lse[1:]).all()
    # For seqlen == 1 (sample 1), softmax weight is identically 1.0 so out[1] == v[cu[1]]
    torch.testing.assert_close(out[1].float(), v[cu[1]].float(), rtol=0, atol=0)
    print("[GOLDEN SELF-CHECK PASS] CPU FP32 reference verified on empty/single/varlen/padded cases.")


# ---------------------------------------------------------------------------
# On-Device Verification Matrix T1-T5 (Spec §7.3)
# ---------------------------------------------------------------------------
def run_device_tests(
    *,
    num_aiv: int,
    block_n: int,
    num_stages: int,
    l2_cache_ctrl: str,
    skip_t5: bool = False,
) -> None:
    """Execute T1-T5 functional, dynamic-shape, boundary, and int64 overflow tests on NPU."""
    device = torch.device("npu")
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    clear_kernel_cache()

    # T1: Baseline case (N=1280, seqlen=578, total_k=739840, seed=7)
    print("\n=== T1: Baseline Case (N=1280, seqlen=578, total_k=739840) ===")
    seqlens_t1 = [BASELINE_SEQLEN] * BASELINE_N
    q_t1, k_t1, v_t1, cu_t1 = make_inputs(seqlens_t1, seed=BASELINE_SEED, device="cpu")
    ref_out_t1, ref_lse_t1 = golden_qsmall_varlen_attn_fwd(q_t1, k_t1, v_t1, cu_t1, sm_scale)

    dev_out_t1, dev_lse_t1 = qsmall_varlen_attn_fwd(
        q_t1.to(device),
        k_t1.to(device),
        v_t1.to(device),
        cu_t1.to(device),
        sm_scale,
        num_aiv=num_aiv,
        block_n=block_n,
        num_stages=num_stages,
        l2_cache_ctrl=l2_cache_ctrl,
    )
    torch.npu.synchronize()
    assert get_compile_count() == 1, f"Expected 1 compile after T1, got {get_compile_count()}"
    check_accuracy("T1_baseline", dev_out_t1, dev_lse_t1, ref_out_t1, ref_lse_t1, seqlens_t1)

    # T2: Dynamic shapes + zero recompilation (also validates 56 AIV vs 64 AIV compatibility!)
    print("\n=== T2: Dynamic Shapes & Multi-Core-Count Zero Recompilation ===")
    t2_configs = [
        ("T2_N1_K578_aiv56", [578], 1.0 / math.sqrt(HEAD_DIM), 56),
        ("T2_N64_K10000_aiv64", [156] * 63 + [172], 0.25, 64),
        ("T2_N1280_K739840_aiv56", [578] * 1280, 1.0 / math.sqrt(HEAD_DIM), 56),
        ("T2_N2048_K500000_aiv64", [244] * 2047 + [532], 0.125, 64),
    ]
    for idx, (label, seqlens, scale, test_aiv) in enumerate(t2_configs):
        q, k, v, cu = make_inputs(seqlens, seed=100 + idx, device="cpu")
        ref_out, ref_lse = golden_qsmall_varlen_attn_fwd(q, k, v, cu, scale)
        dev_out, dev_lse = qsmall_varlen_attn_fwd(
            q.to(device),
            k.to(device),
            v.to(device),
            cu.to(device),
            scale,
            num_aiv=test_aiv,
            block_n=block_n,
            num_stages=num_stages,
            l2_cache_ctrl=l2_cache_ctrl,
        )
        torch.npu.synchronize()
        assert get_compile_count() == 1, f"[{label}] Unexpected recompilation! compile_count={get_compile_count()}"
        check_accuracy(label, dev_out, dev_lse, ref_out, ref_lse, seqlens)

    # T3: Real varlen & boundary lengths (including 0, 1, 2, 15, 31, 32, 33, 63, 64, 65, 578, 1025)
    print("\n=== T3: Real Varlen & Boundary Lengths (incl. empty seqlen=0) ===")
    seqlens_t3 = [0, 1, 2, 15, 31, 32, 33, 63, 64, 65, 578, 1025, 0, 128, 129, 7]
    q_t3, k_t3, v_t3, cu_t3 = make_inputs(seqlens_t3, seed=303, device="cpu")
    ref_out_t3, ref_lse_t3 = golden_qsmall_varlen_attn_fwd(q_t3, k_t3, v_t3, cu_t3, sm_scale)

    # Also test direct kernel call with NaN-prefilled outputs to catch unwritten lanes
    out_nan = torch.full_like(q_t3, float("nan")).to(device)
    lse_nan = torch.full((len(seqlens_t3), NUM_HEADS), float("nan"), dtype=torch.float32).to(device)
    kernel = get_compiled_kernel(block_n=block_n, num_stages=num_stages, l2_cache_ctrl=l2_cache_ctrl)
    grid_n = max(1, min(num_aiv, len(seqlens_t3) * (NUM_HEADS // 8)))
    kernel(q_t3.to(device), k_t3.to(device), v_t3.to(device), cu_t3.to(device), out_nan, lse_nan, float(sm_scale), int(grid_n))
    torch.npu.synchronize()
    assert get_compile_count() == 1
    check_accuracy("T3_varlen_boundaries_nan_sentinel", out_nan, lse_nan, ref_out_t3, ref_lse_t3, seqlens_t3)

    # T4: Uncovered KV buffer (cu_seqlens_k[0] = 17, cu_seqlens_k[N] = total_k - 29)
    print("\n=== T4: Uncovered KV Buffer (prefix_pad=17, suffix_pad=29) ===")
    seqlens_t4 = [0, 13, 64, 129, 578]
    q_t4, k_t4, v_t4, cu_t4 = make_inputs(seqlens_t4, seed=404, prefix_pad=17, suffix_pad=29, device="cpu")
    # Poison unused prefix & suffix regions with NaN to ensure they are never read!
    k_t4[:17] = float("nan")
    k_t4[-29:] = float("nan")
    v_t4[:17] = float("nan")
    v_t4[-29:] = float("nan")
    ref_out_t4, ref_lse_t4 = golden_qsmall_varlen_attn_fwd(q_t4, k_t4, v_t4, cu_t4, sm_scale)
    dev_out_t4, dev_lse_t4 = qsmall_varlen_attn_fwd(
        q_t4.to(device),
        k_t4.to(device),
        v_t4.to(device),
        cu_t4.to(device),
        sm_scale,
        num_aiv=num_aiv,
        block_n=block_n,
        num_stages=num_stages,
        l2_cache_ctrl=l2_cache_ctrl,
    )
    torch.npu.synchronize()
    assert get_compile_count() == 1
    check_accuracy("T4_uncovered_kv_poisoned_guards", dev_out_t4, dev_lse_t4, ref_out_t4, ref_lse_t4, seqlens_t4)

    # T5: int64 offset overflow test (total_k = 1,250,000 -> total_k * 56 * 32 = 2.24e9 > 2^31 - 1)
    if skip_t5:
        print("\n=== T5: SKIPPED (--skip-t5) ===")
    else:
        print("\n=== T5: int64 Offset Overflow (total_k=1,250,000 > int32_max / 1792) ===")
        total_k_t5 = 1_250_000
        # Place samples right past the int32 overflow boundary (element index > 2^31 - 1, row > 1,198,372)
        hi_start = 1_240_000
        seqlens_t5 = [0, 64, 578, 130]
        cu_list_t5 = [hi_start]
        for s in seqlens_t5:
            cu_list_t5.append(cu_list_t5[-1] + s)
        assert cu_list_t5[-1] <= total_k_t5
        assert cu_list_t5[0] * NUM_HEADS * HEAD_DIM > (2**31 - 1)

        gen = torch.Generator(device="cpu").manual_seed(505)
        q_t5 = torch.randn((len(seqlens_t5), NUM_HEADS, HEAD_DIM), generator=gen, dtype=torch.float32).to(torch.bfloat16)
        # Populate only the active high-offset slice on CPU then H2D transfer (avoids NPU-side aclnn ops)
        active_len = cu_list_t5[-1] - hi_start
        k_slice_cpu = torch.randn((active_len, NUM_HEADS, HEAD_DIM), generator=gen, dtype=torch.float32).to(torch.bfloat16)
        v_slice_cpu = torch.randn((active_len, NUM_HEADS, HEAD_DIM), generator=gen, dtype=torch.float32).to(torch.bfloat16)
        k_t5_cpu = torch.empty((total_k_t5, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16)
        v_t5_cpu = torch.empty((total_k_t5, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16)
        k_t5_cpu[hi_start : cu_list_t5[-1]] = k_slice_cpu
        v_t5_cpu[hi_start : cu_list_t5[-1]] = v_slice_cpu
        k_t5_npu = k_t5_cpu.to(device)
        v_t5_npu = v_t5_cpu.to(device)
        del k_t5_cpu, v_t5_cpu

        cu_t5_local = torch.tensor([c - hi_start for c in cu_list_t5], dtype=torch.int32)
        ref_out_t5, ref_lse_t5 = golden_qsmall_varlen_attn_fwd(q_t5, k_slice_cpu, v_slice_cpu, cu_t5_local, sm_scale)

        cu_t5_npu = torch.tensor(cu_list_t5, dtype=torch.int32, device=device)
        dev_out_t5, dev_lse_t5 = qsmall_varlen_attn_fwd(
            q_t5.to(device),
            k_t5_npu,
            v_t5_npu,
            cu_t5_npu,
            sm_scale,
            num_aiv=num_aiv,
            block_n=block_n,
            num_stages=num_stages,
            l2_cache_ctrl=l2_cache_ctrl,
        )
        torch.npu.synchronize()
        assert get_compile_count() == 1
        check_accuracy("T5_int64_overflow_1.25M", dev_out_t5, dev_lse_t5, ref_out_t5, ref_lse_t5, seqlens_t5)
        del k_t5_npu, v_t5_npu


# ---------------------------------------------------------------------------
# Performance Benchmark (Spec §7.4: warmup=10, iters=50, median ms & GB/s)
# ---------------------------------------------------------------------------
def benchmark_baseline(
    *,
    num_aiv: int,
    block_n: int,
    num_stages: int,
    l2_cache_ctrl: str,
    warmup: int = 10,
    iters: int = 50,
) -> tuple[float, float]:
    """Benchmark baseline shape (N=1280, seqlen=578, total_k=739840) per Spec §7.4."""
    device = torch.device("npu")
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    seqlens = [BASELINE_SEQLEN] * BASELINE_N
    q, k, v, cu = make_inputs(seqlens, seed=BASELINE_SEED, device=device)

    for _ in range(warmup):
        qsmall_varlen_attn_fwd(
            q,
            k,
            v,
            cu,
            sm_scale,
            num_aiv=num_aiv,
            block_n=block_n,
            num_stages=num_stages,
            l2_cache_ctrl=l2_cache_ctrl,
        )
    torch.npu.synchronize()

    times_ms: list[float] = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        qsmall_varlen_attn_fwd(
            q,
            k,
            v,
            cu,
            sm_scale,
            num_aiv=num_aiv,
            block_n=block_n,
            num_stages=num_stages,
            l2_cache_ctrl=l2_cache_ctrl,
        )
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    median_ms = statistics.median(times_ms)
    min_ms = min(times_ms)
    eff_bw_gbs = BASELINE_TOTAL_BYTES_GB / (median_ms / 1000.0)
    print(
        f"[BENCH] num_aiv={num_aiv} block_n={block_n} stages={num_stages} l2={l2_cache_ctrl:<14s} | "
        f"median={median_ms:.3f} ms (min={min_ms:.3f} ms) | BW_eff={eff_bw_gbs:.1f} GB/s"
    )
    return median_ms, eff_bw_gbs


def run_profile_baseline(
    num_aiv: int,
    block_n: int,
    num_stages: int,
    l2_cache_ctrl: str,
    iters: int = 5,
) -> None:
    """Lightweight baseline-only run (N=1280, L=578) tailored for msprof / msprof op.

    Initializes inputs on CPU and copies to NPU via H2D so toolkit-only CANN
    environments without OPP random kernels work reliably.
    """
    torch.manual_seed(7)
    n, seqlen = 1280, 578
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    q = torch.randn((n, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16).to("npu")
    k = torch.randn((seqlen, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16).repeat(n, 1, 1).contiguous().to("npu")
    v = torch.randn((seqlen, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16).repeat(n, 1, 1).contiguous().to("npu")
    cu = (torch.arange(n + 1, dtype=torch.int32) * seqlen).to("npu")
    torch.npu.synchronize()

    for _ in range(iters):
        qsmall_varlen_attn_fwd(
            q,
            k,
            v,
            cu,
            sm_scale,
            num_aiv=num_aiv,
            block_n=block_n,
            num_stages=num_stages,
            l2_cache_ctrl=l2_cache_ctrl,
        )
    torch.npu.synchronize()
    print(
        f"[PROFILE] Completed {iters} baseline launches "
        f"(N={n}, L={seqlen}, num_aiv={num_aiv}, block_n={block_n}, stages={num_stages}, l2={l2_cache_ctrl})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Offline mode (no NPU): compile kernel, statically audit CCE, and self-check CPU Golden.",
    )
    parser.add_argument("--num-aiv", type=int, default=None, help="Override AIV core count (e.g. 56 for 950PR, 64 for 950DT).")
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N, help="KV block size (default: 64).")
    parser.add_argument("--num-stages", type=int, default=DEFAULT_NUM_STAGES, help="Pipeline stages (default: 3).")
    parser.add_argument("--l2-cache-ctrl", type=str, default=DEFAULT_L2_CACHE_CTRL, help="L2 cache policy for K/V loads.")
    parser.add_argument("--dump-cce", type=str, default=None, help="Optional path to dump generated CCE source.")
    parser.add_argument("--skip-t5", action="store_true", help="Skip 1.25M int64 overflow test (saves ~9 GB HBM).")
    parser.add_argument("--bench", action="store_true", help="Run baseline latency & effective HBM bandwidth benchmark.")
    parser.add_argument("--sweep", action="store_true", help="Sweep tuning configs (stages, l2_cache_ctrl, block_n) on NPU.")
    parser.add_argument("--warmup", type=int, default=10, help="Benchmark warmup iterations (Spec §7.4 default: 10).")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark timed iterations (Spec §7.4 default: 50).")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Lightweight baseline-only run (skip T1-T5 & CCE audit) for msprof / msprof op.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=5,
        help="Number of baseline kernel launches in --profile-only mode (default: 5).",
    )
    args = parser.parse_args()

    if args.compile_only:
        print("=== Offline Validation Mode (Compile + CCE Audit + CPU Golden) ===")
        audit_compiled_cce(args.block_n, args.num_stages, args.l2_cache_ctrl, dump_path=args.dump_cce)
        run_cpu_golden_self_check()
        print("\nAll offline checks PASSED. On-device NPU accuracy/timing: not_run (--compile-only).")
        return

    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        raise RuntimeError(
            "No physical Ascend NPU detected (`torch.npu.is_available()` is False). "
            "Run with `--compile-only` for local WSL compilation & CCE audit."
        )

    active_aiv = detect_num_aiv() if args.num_aiv is None else args.num_aiv
    dev_name = npu.get_device_name(npu.current_device()) if hasattr(npu, "get_device_name") else "Ascend NPU"
    print(f"=== Ascend NPU Verification: device={dev_name}, active_aiv={active_aiv} ===")

    if args.profile_only:
        run_profile_baseline(
            num_aiv=active_aiv,
            block_n=args.block_n,
            num_stages=args.num_stages,
            l2_cache_ctrl=args.l2_cache_ctrl,
            iters=args.profile_iters,
        )
        return

    audit_compiled_cce(args.block_n, args.num_stages, args.l2_cache_ctrl, dump_path=args.dump_cce)
    run_device_tests(
        num_aiv=active_aiv,
        block_n=args.block_n,
        num_stages=args.num_stages,
        l2_cache_ctrl=args.l2_cache_ctrl,
        skip_t5=args.skip_t5,
    )

    if args.bench:
        print(f"\n=== Baseline Benchmark (warmup={args.warmup}, iters={args.iters}) ===")
        benchmark_baseline(
            num_aiv=active_aiv,
            block_n=args.block_n,
            num_stages=args.num_stages,
            l2_cache_ctrl=args.l2_cache_ctrl,
            warmup=args.warmup,
            iters=args.iters,
        )

    if args.sweep:
        print("\n=== Tuning Sweep on Baseline Shape ===")
        sweep_configs = [
            (64, 3, "NOTALLOC_KEEP"),
            (64, 2, "NOTALLOC_KEEP"),
            (48, 4, "NOTALLOC_KEEP"),
            (64, 3, "NORMAL_FV"),
            (32, 3, "NOTALLOC_KEEP"),
        ]
        for bn, st, l2 in sweep_configs:
            benchmark_baseline(
                num_aiv=active_aiv,
                block_n=bn,
                num_stages=st,
                l2_cache_ctrl=l2,
                warmup=args.warmup,
                iters=args.iters,
            )


if __name__ == "__main__":
    main()
