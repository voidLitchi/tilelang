"""Verification & NPU accuracy/benchmark suite for Lightning Indexer Sparse Backward.

All NPU inputs and NaN output sentinels are constructed on CPU and transferred
via ``.to("npu")`` (pure H2D ``aclrtMemcpy``) so the script works cleanly even
on toolkit-only CANN 9.3.0-beta.1 installs that omit selected ``aclnn`` OPP ops.

Usage examples
--------------
1. CPU reference & FP64 gradcheck (no NPU required):
   ``python examples/ascend/lightning_indexer/verify_lightning_indexer_bwd.py --mode cpu-ref``

2. Local/remote offline CANN compilation & CCE audit (no NPU lock needed with TORCH_DEVICE_BACKEND_AUTOLOAD=0):
   ``python examples/ascend/lightning_indexer/verify_lightning_indexer_bwd.py --mode compile --num-cores 32``
   ``python examples/ascend/lightning_indexer/verify_lightning_indexer_bwd.py --mode compile --num-cores 28``

3. Full end-to-end Ascend 950DT NPU accuracy verification (wrap in ``npu-guard run``):
   ``python examples/ascend/lightning_indexer/verify_lightning_indexer_bwd.py --mode npu --num-cores 32 --bk 64``
   ``python examples/ascend/lightning_indexer/verify_lightning_indexer_bwd.py --mode npu --num-cores 28 --bk 128``

4. NPU accuracy + AB/BA paired benchmark (comparing BK=64/128 and staged vs unstaged):
   ``python examples/ascend/lightning_indexer/verify_lightning_indexer_bwd.py --mode npu --num-cores 32 --bench``

5. Single-operator run-only mode without accuracy verification (for ``msprof op`` hardware pipeline capture):
   ``msprof op --output=./msprof_li_bwd -- python examples/ascend/lightning_indexer/verify_lightning_indexer_bwd.py --mode profile --num-cores 28 --bk 128 --only-main-kernel``
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import torch

from examples.ascend.lightning_indexer.reference import (
    SPEC_THRESHOLDS,
    compute_accuracy_metrics,
    ref_lightning_indexer_sparse_bwd,
    ref_lightning_indexer_sparse_bwd_analytical,
    ref_lightning_indexer_sparse_fwd,
)


def make_unique_topk_indices(
    B: int,
    S_q: int,
    S_k: int,
    K: int,
    *,
    valid_counts: torch.Tensor | None = None,
    sink_tokens: int = 0,
    seed: int = 42,
    shuffle_invalid: bool = False,
) -> torch.Tensor:
    """Generate unique per-query Top-K indices in ``[0, S_k)`` padded with ``-1`` on CPU."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    indices = torch.full((B, S_q, K), -1, dtype=torch.int32)
    if valid_counts is None:
        v_counts = torch.full((B, S_q), min(K, S_k), dtype=torch.int32)
    else:
        v_counts = valid_counts.to("cpu", dtype=torch.int32)

    if S_k <= 0:
        return indices

    for b in range(B):
        for t in range(S_q):
            cnt = min(int(v_counts[b, t].item()), K, S_k)
            if cnt <= 0:
                continue
            if sink_tokens > 0 and S_k >= sink_tokens:
                n_sink = min(sink_tokens, cnt)
                row_vals = list(range(n_sink))
                rem = cnt - n_sink
                if rem > 0 and S_k > n_sink:
                    pool = torch.randperm(S_k - n_sink, generator=gen)[:rem] + n_sink
                    row_vals.extend(pool.tolist())
                chosen = torch.tensor(row_vals, dtype=torch.int32)
            else:
                chosen = torch.randperm(S_k, generator=gen)[:cnt].to(torch.int32)
            indices[b, t, :cnt] = chosen
            if shuffle_invalid and cnt < K:
                perm = torch.randperm(K, generator=gen)
                indices[b, t] = indices[b, t, perm]

    return indices


def make_fast_large_indices(
    B: int,
    S_q: int,
    S_k: int,
    K: int,
    *,
    sink_tokens: int = 0,
    seed: int = 42,
) -> torch.Tensor:
    """Vectorized CPU generator for large shapes (e.g., S_q=32768, S_k=16384, K=512)."""
    del seed  # deterministic pattern with high adjacent-query overlap
    valid_k = min(K, S_k)
    indices = torch.full((B, S_q, K), -1, dtype=torch.int32)
    if valid_k == 0:
        return indices

    q_pos = torch.arange(S_q, dtype=torch.int64).view(1, S_q, 1)
    k_pos = torch.arange(valid_k, dtype=torch.int64).view(1, 1, valid_k)
    b_off = (torch.arange(B, dtype=torch.int64) * 131).view(B, 1, 1)

    if sink_tokens > 0 and S_k > sink_tokens and valid_k > sink_tokens:
        non_sink_sk = S_k - sink_tokens
        rem_k = valid_k - sink_tokens
        rem_pos = torch.arange(rem_k, dtype=torch.int64).view(1, 1, rem_k)
        base = (q_pos // 2) + b_off
        tail = sink_tokens + ((base + rem_pos) % non_sink_sk)
        indices[:, :, :sink_tokens] = torch.arange(sink_tokens, dtype=torch.int32).view(1, 1, sink_tokens)
        indices[:, :, sink_tokens:valid_k] = tail.to(torch.int32)
    else:
        base = (q_pos // 2) + b_off
        vals = (base + k_pos) % S_k
        indices[:, :, :valid_k] = vals.to(torch.int32)

    return indices


def make_scattered_large_indices(
    B: int,
    S_q: int,
    S_k: int,
    K: int,
    *,
    sink_tokens: int = 4,
    stride: int = 29,
) -> torch.Tensor:
    """Vectorized CPU generator for worst-case NON-contiguous scattered Top-K indices.

    Uses a prime stride coprime to power-of-two ``S_k`` (or ``S_k - sink_tokens``)
    so every row has strictly unique indices with ZERO contiguous runs of length >= 2.
    """
    valid_k = min(K, S_k)
    indices = torch.full((B, S_q, K), -1, dtype=torch.int32)
    if valid_k == 0:
        return indices

    q_pos = torch.arange(S_q, dtype=torch.int64).view(1, S_q, 1)
    b_off = (torch.arange(B, dtype=torch.int64) * 131).view(B, 1, 1)

    if sink_tokens > 0 and S_k > sink_tokens and valid_k > sink_tokens:
        non_sink_sk = S_k - sink_tokens
        st = stride
        while math.gcd(st, non_sink_sk) != 1:
            st += 2
        rem_k = valid_k - sink_tokens
        rem_pos = torch.arange(rem_k, dtype=torch.int64).view(1, 1, rem_k)
        tail = sink_tokens + (((q_pos * 7 + b_off) + rem_pos * st) % non_sink_sk)
        indices[:, :, :sink_tokens] = torch.arange(sink_tokens, dtype=torch.int32).view(1, 1, sink_tokens)
        indices[:, :, sink_tokens:valid_k] = tail.to(torch.int32)
    else:
        st = stride
        while math.gcd(st, S_k) != 1:
            st += 2
        k_pos = torch.arange(valid_k, dtype=torch.int64).view(1, 1, valid_k)
        vals = ((q_pos * 7 + b_off) + k_pos * st) % S_k
        indices[:, :, :valid_k] = vals.to(torch.int32)

    return indices


def run_cpu_reference_checks() -> None:
    """Validate PyTorch Autograd reference, analytical reference, and FP64 gradcheck on CPU."""
    print("=== [1/2] FP64 torch.autograd.gradcheck on ref_lightning_indexer_sparse_fwd ===")
    torch.manual_seed(1234)
    B, S_q, S_k, H, D, K = 2, 4, 6, 3, 5, 8
    q64 = torch.randn(B, S_q, H, D, dtype=torch.float64, requires_grad=True)
    k64 = torch.randn(B, S_k, D, dtype=torch.float64, requires_grad=True)
    w64 = torch.randn(B, S_q, H, dtype=torch.float64, requires_grad=True)
    indices = torch.tensor(
        [
            [
                [0, 1, 2, 3, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [0, 1, 5, -1, 2, 99, -1, 4],
                [0, 1, 2, 3, 4, 5, -1, -1],
            ],
            [
                [0, 1, 2, -1, -1, -1, -1, -1],
                [5, 4, 3, 2, 1, 0, -1, -1],
                [0, 1, -1, -1, -1, -1, -1, -1],
                [3, 1, 4, 0, -1, -1, -1, -1],
            ],
        ],
        dtype=torch.int32,
    )
    scale = 1.0 / math.sqrt(D)
    passed = torch.autograd.gradcheck(
        lambda q_, k_, w_: ref_lightning_indexer_sparse_fwd(q_, k_, w_, indices, scale=scale),
        (q64, k64, w64),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
    assert passed, "FP64 gradcheck failed!"
    print("  PASS: FP64 gradcheck passed across valid, all-(-1), scattered -1, OOB, and sink-overlap rows.")

    print("=== [2/2] Cross-check Autograd FP32 ref vs Analytical FP32 ref across edge cases ===")
    cases = [
        ("basic_v41_small", dict(B=2, S_q=16, S_k=640, H=32, D=128, K=512, scale=1.0 / math.sqrt(128))),
        ("short_sk_lt_k", dict(B=1, S_q=12, S_k=128, H=32, D=128, K=512, scale=1.0)),
        ("all_invalid_and_partial_73", dict(B=1, S_q=8, S_k=300, H=32, D=128, K=512, scale=0.125)),
        ("sink_tokens", dict(B=1, S_q=16, S_k=600, H=32, D=128, K=512, scale=1.0)),
        ("all_negative_dot", dict(B=1, S_q=8, S_k=600, H=32, D=128, K=512, scale=1.0)),
    ]
    for name, cfg in cases:
        torch.manual_seed(100)
        B, S_q, S_k, H, D, K, scale = cfg["B"], cfg["S_q"], cfg["S_k"], cfg["H"], cfg["D"], cfg["K"], cfg["scale"]
        q = torch.randn(B, S_q, H, D, dtype=torch.bfloat16)
        k = torch.randn(B, S_k, D, dtype=torch.bfloat16)
        w = torch.randn(B, S_q, H, dtype=torch.float32)
        dI = torch.randn(B, S_q, K, dtype=torch.float32)

        if name == "all_invalid_and_partial_73":
            vc = torch.tensor([[0, 73, 1, 64, 65, 128, 129, 0]], dtype=torch.int32)
            idx = make_unique_topk_indices(B, S_q, S_k, K, valid_counts=vc, seed=7)
            dI = torch.where(idx < 0, torch.full_like(dI, float("nan")), dI)
        elif name == "sink_tokens":
            idx = make_unique_topk_indices(B, S_q, S_k, K, sink_tokens=4, seed=9)
        elif name == "all_negative_dot":
            q = torch.abs(q) + 0.5
            k = -(torch.abs(k) + 0.5)
            idx = make_unique_topk_indices(B, S_q, S_k, K, seed=11)
        else:
            idx = make_unique_topk_indices(B, S_q, S_k, K, seed=42)

        dI_clean = torch.where((idx >= 0) & (idx < S_k), dI, torch.zeros_like(dI))
        dq_auto, dk_auto, dw_auto = ref_lightning_indexer_sparse_bwd(q, k, w, idx, dI_clean, scale=scale, q_chunk_size=4)
        dq_ana, dk_ana, dw_ana = ref_lightning_indexer_sparse_bwd_analytical(q, k, w, idx, dI_clean, scale=scale, q_chunk_size=4)

        for tname, a, b in (("dq", dq_auto, dq_ana), ("dk", dk_auto, dk_ana), ("dw", dw_auto, dw_ana)):
            m = compute_accuracy_metrics(tname, a, b)
            assert not m.has_nan and not m.has_inf, f"{name}/{tname} has NaN/Inf!"
            assert m.max_abs_error < 1e-4 and m.rel_l2_error < 1e-5, (
                f"{name}/{tname} mismatch: max_abs={m.max_abs_error}, rel_l2={m.rel_l2_error}"
            )
            if name == "all_negative_dot":
                assert a.abs().max().item() == 0.0 and b.abs().max().item() == 0.0

        if name == "all_invalid_and_partial_73":
            assert dq_auto[0, 0].abs().max().item() == 0.0 and dw_auto[0, 0].abs().max().item() == 0.0
            assert dq_auto[0, 7].abs().max().item() == 0.0 and dw_auto[0, 7].abs().max().item() == 0.0
        print(f"  PASS: {name} (autograd vs analytical exact match)")


def run_compile_checks(
    *,
    num_cores_list: tuple[int, ...] = (28, 32, 36),
    staged_pipeline: bool = True,
    interleaved_pipeline: bool = True,
    dump_source_dir: str | None = None,
) -> None:
    """Compile TileLang Ascend kernels to .o and audit generated CCE source (no NPU needed)."""
    import tilelang
    from examples.ascend.lightning_indexer.kernel import (
        build_cast_fp32_to_bf16_kernel,
        build_lightning_indexer_sparse_bwd_kernel,
        build_zero_fp32_kernel,
    )

    out_dir = Path(dump_source_dir) if dump_source_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Compile auxiliary zero & cast SIMD kernels once per core count.
    for nc in num_cores_list:
        for aux_name, aux_prim in (
            (f"zero_fp32_c{nc}", build_zero_fp32_kernel(D=128, num_cores=nc)),
            (f"cast_fp32_to_bf16_c{nc}", build_cast_fp32_to_bf16_kernel(D=128, num_cores=nc)),
        ):
            k_aux = tilelang.compile(aux_prim, target="ascend", out_idx=[])
            aux_src = k_aux.get_kernel_source()
            assert "__simt__" not in aux_src and "asc_syncthreads" not in aux_src
            print(f"  PASS aux {aux_name}: compiled cleanly ({len(aux_src)} chars, pure SIMD)")

    mode_tag = (
        "v2_interleaved"
        if (staged_pipeline and interleaved_pipeline)
        else ("v1_staged" if staged_pipeline else "flat")
    )
    configs = []
    for nc in num_cores_list:
        for bk in (64, 128):
            configs.append(
                (
                    f"c{nc}_bk{bk}_{mode_tag}_early_exit",
                    dict(
                        BK=bk,
                        num_cores=nc,
                        sorted_invalid_tail=True,
                        staged_pipeline=staged_pipeline,
                        interleaved_pipeline=interleaved_pipeline,
                    ),
                )
            )
    # Also verify unsorted path on acceptance standard (32 AIC)
    if 32 in num_cores_list and not staged_pipeline:
        for bk in (64, 128):
            configs.append(
                (
                    f"c32_bk{bk}_{mode_tag}_unsorted",
                    dict(
                        BK=bk,
                        num_cores=32,
                        sorted_invalid_tail=False,
                        staged_pipeline=staged_pipeline,
                        interleaved_pipeline=interleaved_pipeline,
                    ),
                )
            )

    for label, cfg in configs:
        print(f"=== Compiling Ascend kernel: {label} ({cfg}) ===")
        prim = build_lightning_indexer_sparse_bwd_kernel(H=32, D=128, K=512, **cfg)
        kernel = tilelang.compile(prim, target="ascend", out_idx=[])
        src = kernel.get_kernel_source()
        assert src and len(src) > 100, f"{label}: empty kernel source!"

        has_simt = ("__simt__" in src) or ("asc_syncthreads" in src)
        assert not has_simt, f"{label}: unexpected SIMT codegen detected!"
        assert "asc_set_atomic_add_float" in src, f"{label}: missing hardware FP32 atomic add!"
        assert "asc_set_atomic_none" in src, f"{label}: missing asc_set_atomic_none restore!"
        assert "vcadd" in src, f"{label}: missing SIMD vcadd reduction!"
        assert "vcvt" in src, f"{label}: missing SIMD vcvt cast!"
        if cfg["BK"] == 128:
            assert "vsstb" in src, f"{label}: missing compact-NZ SIMD vsstb packing!"

        if out_dir is not None:
            path = out_dir / f"lightning_indexer_sparse_bwd_{label}.cpp"
            path.write_text(src, encoding="utf-8")
            print(f"  Dumped source -> {path} ({len(src)} bytes)")
        print(
            f"  PASS: {label} compiled cleanly ({len(src)} chars, zero SIMT, SIMD + Cube + MTE3 atomic verified)"
        )


def _check_case_npu(
    case_name: str,
    q_cpu: torch.Tensor,
    k_cpu: torch.Tensor,
    w_cpu: torch.Tensor,
    idx_cpu: torch.Tensor,
    dI_cpu: torch.Tensor,
    *,
    scale: float = 1.0,
    sorted_invalid_tail: bool = True,
    bk: int = 128,
    num_cores: int = 32,
    staged_pipeline: bool = True,
    interleaved_pipeline: bool = True,
    dk_l2_policy: str = "NORMAL_FV",
    k_l2_policy: str = "NORMAL_FV",
    use_analytical_ref: bool = False,
    expect_all_zero: bool = False,
    zero_query_rows: list[tuple[int, int]] | None = None,
) -> None:
    """Run one NPU accuracy case using CPU-built inputs & NaN sentinels (toolkit-only CANN safe)."""
    from examples.ascend.lightning_indexer.kernel import lightning_indexer_sparse_bwd

    q = q_cpu.to("npu")
    k = k_cpu.to("npu")
    w = w_cpu.to("npu")
    indices = idx_cpu.to("npu")
    dI = dI_cpu.to("npu")

    # Eval Invariant #1: pre-fill ALL output buffers with NaN sentinels on CPU -> H2D.
    dq_out = torch.full(q_cpu.shape, float("nan"), dtype=torch.bfloat16).to("npu")
    dk_out = torch.full(k_cpu.shape, float("nan"), dtype=torch.bfloat16).to("npu")
    dw_out = torch.full(w_cpu.shape, float("nan"), dtype=torch.float32).to("npu")
    dk_ws = torch.full(k_cpu.shape, float("nan"), dtype=torch.float32).to("npu")

    dq, dk, dw = lightning_indexer_sparse_bwd(
        q,
        k,
        w,
        indices,
        dI,
        scale=scale,
        dk_workspace=dk_ws,
        sorted_invalid_tail=sorted_invalid_tail,
        bk=bk,
        num_cores=num_cores,
        staged_pipeline=staged_pipeline,
        interleaved_pipeline=interleaved_pipeline,
        dk_l2_policy=dk_l2_policy,
        k_l2_policy=k_l2_policy,
        dq_out=dq_out,
        dk_out=dk_out,
        dw_out=dw_out,
    )
    torch.npu.synchronize()

    dq_res_cpu = dq.to("cpu")
    dk_res_cpu = dk.to("cpu")
    dw_res_cpu = dw.to("cpu")

    # Compute independent FP32 reference on CPU (always exact, zero OPP op risk).
    valid = (idx_cpu >= 0) & (idx_cpu < k_cpu.shape[1])
    dI_ref = torch.where(valid, dI_cpu, torch.zeros_like(dI_cpu))
    if use_analytical_ref:
        dq_ref, dk_ref, dw_ref = ref_lightning_indexer_sparse_bwd_analytical(
            q_cpu, k_cpu, w_cpu, idx_cpu, dI_ref, scale=scale, q_chunk_size=128
        )
    else:
        dq_ref, dk_ref, dw_ref = ref_lightning_indexer_sparse_bwd(
            q_cpu, k_cpu, w_cpu, idx_cpu, dI_ref, scale=scale, q_chunk_size=128
        )

    print(
        f"\n[Case: {case_name}] B={q_cpu.shape[0]} S_q={q_cpu.shape[1]} S_k={k_cpu.shape[1]} "
        f"H={q_cpu.shape[2]} D={q_cpu.shape[3]} K={idx_cpu.shape[2]} "
        f"cores={num_cores} bk={bk} staged={staged_pipeline} scale={scale:.4g}"
    )
    for name, act, ref in (("dq", dq_res_cpu, dq_ref), ("dk", dk_res_cpu, dk_ref), ("dw", dw_res_cpu, dw_ref)):
        m = compute_accuracy_metrics(name, act, ref)
        th = SPEC_THRESHOLDS[name]
        ok = m.passes(min_cos=th["min_cos"], max_rel_l2=th["max_rel_l2"])
        status = "PASS" if ok else "FAIL"
        print(
            f"  {status} {name:2s} ({m.dtype:14s}): "
            f"cos={m.cosine_similarity:.6f} (>={th['min_cos']}), "
            f"rel_l2={m.rel_l2_error:.6e} (<={th['max_rel_l2']:.1e}), "
            f"p99.9={m.p999_abs_error:.4e}, max_abs={m.max_abs_error:.4e}, "
            f"nan={m.has_nan}, inf={m.has_inf}"
        )
        assert ok, f"Accuracy check FAILED for {case_name}/{name}: {m}"
        if expect_all_zero:
            assert m.max_abs_error == 0.0, f"{case_name}/{name}: expected strict zeros, got max_abs={m.max_abs_error}"

    if zero_query_rows:
        for b_i, t_i in zero_query_rows:
            assert dq_res_cpu[b_i, t_i].float().abs().max().item() == 0.0, (
                f"{case_name}: all-(-1) row ({b_i},{t_i}) dq non-zero!"
            )
            assert dw_res_cpu[b_i, t_i].float().abs().max().item() == 0.0, (
                f"{case_name}: all-(-1) row ({b_i},{t_i}) dw non-zero!"
            )


def run_npu_verification(
    *,
    bk: int = 128,
    num_cores: int = 32,
    staged_pipeline: bool = True,
    interleaved_pipeline: bool = True,
    dk_l2_policy: str = "NORMAL_FV",
    k_l2_policy: str = "NORMAL_FV",
    include_large: bool = True,
    bench: bool = False,
) -> None:
    """Run all Spec Section 6.2 & Eval-Invariant checks on Ascend 950DT NPU."""
    import torch_npu  # noqa: F401
    from examples.ascend.lightning_indexer.kernel import (
        get_compile_count,
        lightning_indexer_sparse_bwd,
        reset_compile_cache,
    )

    H, D, K = 32, 128, 512
    print(
        f"=== Running NPU Accuracy Suite (num_cores={num_cores} AIC / {num_cores*2} AIV, "
        f"BK={bk}, staged={staged_pipeline}, interleaved={interleaved_pipeline}, "
        f"k_l2={k_l2_policy}, dk_l2={dk_l2_policy}) ==="
    )
    reset_compile_cache()

    common_kw = dict(
        bk=bk,
        num_cores=num_cores,
        staged_pipeline=staged_pipeline,
        interleaved_pipeline=interleaved_pipeline,
        dk_l2_policy=dk_l2_policy,
        k_l2_policy=k_l2_policy,
    )

    # 1. Basic multi-batch + non-unit scale (gold-standard autograd ref)
    torch.manual_seed(42)
    B, S_q, S_k = 2, 64, 1024
    q = torch.randn(B, S_q, H, D, dtype=torch.bfloat16)
    k = torch.randn(B, S_k, D, dtype=torch.bfloat16)
    w = torch.randn(B, S_q, H, dtype=torch.float32)
    dI = torch.randn(B, S_q, K, dtype=torch.float32)
    idx = make_unique_topk_indices(B, S_q, S_k, K, seed=42)
    _check_case_npu("basic_batched_scaled", q, k, w, idx, dI, scale=1.0 / math.sqrt(D), **common_kw)

    # 2. Short sequence S_k = 128 < K = 512 (heavy -1 padding)
    B, S_q, S_k = 1, 64, 128
    q = torch.randn(B, S_q, H, D, dtype=torch.bfloat16)
    k = torch.randn(B, S_k, D, dtype=torch.bfloat16)
    w = torch.randn(B, S_q, H, dtype=torch.float32)
    dI = torch.randn(B, S_q, K, dtype=torch.float32)
    idx = make_unique_topk_indices(B, S_q, S_k, K, seed=43)
    _check_case_npu("short_seq_sk128_lt_k512", q, k, w, idx, dI, scale=1.0, **common_kw)

    # 3. All-(-1) rows + mixed partial non-BK-aligned valid prefix (73 valid, 439 -1) + NaN in invalid dI
    #    Note: S_q=16 < num_cores (28/32/36) also exercises q_count==0 & q_count==1 cores!
    B, S_q, S_k = 1, 16, 600
    q = torch.randn(B, S_q, H, D, dtype=torch.bfloat16)
    k = torch.randn(B, S_k, D, dtype=torch.bfloat16)
    w = torch.randn(B, S_q, H, dtype=torch.float32)
    dI = torch.randn(B, S_q, K, dtype=torch.float32)
    vc = torch.tensor([[0, 73, 1, 63, 64, 65, 127, 128, 129, 0, 512, 255, 256, 333, 511, 0]], dtype=torch.int32)
    idx = make_unique_topk_indices(B, S_q, S_k, K, valid_counts=vc, seed=44)
    dI_nan = torch.where(idx < 0, torch.full_like(dI, float("nan")), dI)
    _check_case_npu(
        "all_neg1_and_partial_73", q, k, w, idx, dI_nan,
        scale=1.0, sorted_invalid_tail=True, zero_query_rows=[(0, 0), (0, 9), (0, 15)], **common_kw,
    )

    # 4. Unsorted invalid (-1 and OOB >= S_k) with sorted_invalid_tail=False
    idx_unsorted = make_unique_topk_indices(B, S_q, S_k, K, valid_counts=vc, seed=45, shuffle_invalid=True)
    idx_unsorted[0, 1, 0] = S_k + 17
    _check_case_npu(
        "unsorted_invalid_and_oob", q, k, w, idx_unsorted, dI,
        scale=0.5, sorted_invalid_tail=False, zero_query_rows=[(0, 0), (0, 9), (0, 15)], **common_kw,
    )

    # 5. High-contention Sink Token Stress Test (indices[..., 0:4] = [0, 1, 2, 3])
    B, S_q, S_k = 1, 1024, 2048
    q = torch.randn(B, S_q, H, D, dtype=torch.bfloat16)
    k = torch.randn(B, S_k, D, dtype=torch.bfloat16)
    w = torch.randn(B, S_q, H, dtype=torch.float32)
    dI = torch.randn(B, S_q, K, dtype=torch.float32)
    idx = make_fast_large_indices(B, S_q, S_k, K, sink_tokens=4, seed=46)
    _check_case_npu(
        "sink_token_stress", q, k, w, idx, dI,
        scale=1.0 / math.sqrt(D), use_analytical_ref=True, **common_kw,
    )

    # 6. All-negative dot products (q > 0, k < 0 -> strict all-zero outputs)
    B, S_q, S_k = 1, 64, 1024
    q_pos = torch.abs(torch.randn(B, S_q, H, D, dtype=torch.bfloat16)) + 0.5
    k_neg = -(torch.abs(torch.randn(B, S_k, D, dtype=torch.bfloat16)) + 0.5)
    w = torch.randn(B, S_q, H, dtype=torch.float32)
    dI = torch.randn(B, S_q, K, dtype=torch.float32)
    idx = make_unique_topk_indices(B, S_q, S_k, K, seed=47)
    _check_case_npu(
        "all_negative_dot", q_pos, k_neg, w, idx, dI,
        scale=1.0, expect_all_zero=True, **common_kw,
    )

    # 7. Dynamic shape NO-RECOMPILE verification (Spec Section 6.2 #5)
    print("\n=== [Case: dynamic_shape_no_recompile] ===")
    reset_compile_cache()
    for s_q, s_k in ((1024, 512), (4096, 2048), (8193, 4097), (16384, 8192)):
        q_d = torch.randn(1, s_q, H, D, dtype=torch.bfloat16).to("npu")
        k_d = torch.randn(1, s_k, D, dtype=torch.bfloat16).to("npu")
        w_d = torch.randn(1, s_q, H, dtype=torch.float32).to("npu")
        dI_d = torch.randn(1, s_q, K, dtype=torch.float32).to("npu")
        idx_d = make_fast_large_indices(1, s_q, s_k, K, seed=s_q + s_k).to("npu")
        dq_d, dk_d, dw_d = lightning_indexer_sparse_bwd(
            q_d, k_d, w_d, idx_d, dI_d, scale=1.0, **common_kw,
        )
        torch.npu.synchronize()
        assert get_compile_count() == 1, f"Unexpected recompile! compile_count={get_compile_count()} at ({s_q}, {s_k})"
        assert not torch.isnan(dq_d.to("cpu")).any()
        assert not torch.isnan(dk_d.to("cpu")).any()
        assert not torch.isnan(dw_d.to("cpu")).any()
        print(f"  PASS dynamic shape (S_q={s_q}, S_k={s_k}) | compile_count={get_compile_count()}")

    # 8. In-place mutation of indices/dI at same device pointer (Eval Invariant #3) & accumulate_dk
    print("\n=== [Case: inplace_mutation_and_accumulate_dk] ===")
    B, S_q, S_k = 1, 32, 600
    q_m_cpu = torch.randn(B, S_q, H, D, dtype=torch.bfloat16)
    k_m_cpu = torch.randn(B, S_k, D, dtype=torch.bfloat16)
    w_m_cpu = torch.randn(B, S_q, H, dtype=torch.float32)
    dI_m_cpu = torch.randn(B, S_q, K, dtype=torch.float32)
    idx_m_cpu = make_unique_topk_indices(B, S_q, S_k, K, seed=80)

    q_m = q_m_cpu.to("npu")
    k_m = k_m_cpu.to("npu")
    w_m = w_m_cpu.to("npu")
    dI_m = dI_m_cpu.to("npu")
    idx_m = idx_m_cpu.to("npu")
    ws = torch.zeros(B, S_k, D, dtype=torch.float32).to("npu")

    dq1, dk1, dw1 = lightning_indexer_sparse_bwd(q_m, k_m, w_m, idx_m, dI_m, dk_workspace=ws, **common_kw)
    torch.npu.synchronize()
    dk1_cpu = dk1.to("cpu")

    # Mutate indices & dI in-place via CPU -> NPU H2D copy into the SAME device buffers
    idx_m2_cpu = make_unique_topk_indices(B, S_q, S_k, K, seed=81)
    dI_m2_cpu = dI_m_cpu * (-1.5)
    idx_m.copy_(idx_m2_cpu)
    dI_m.copy_(dI_m2_cpu)

    dq2, dk_acc, dw2 = lightning_indexer_sparse_bwd(
        q_m, k_m, w_m, idx_m, dI_m, dk_workspace=ws, accumulate_dk=True, **common_kw,
    )
    torch.npu.synchronize()
    _, dk2_ref, _ = ref_lightning_indexer_sparse_bwd(q_m_cpu, k_m_cpu, w_m_cpu, idx_m2_cpu, dI_m2_cpu, scale=1.0)
    m_acc = compute_accuracy_metrics("dk_accumulated", dk_acc.to("cpu"), dk1_cpu.float() + dk2_ref)
    assert m_acc.passes(min_cos=SPEC_THRESHOLDS["dk"]["min_cos"], max_rel_l2=SPEC_THRESHOLDS["dk"]["max_rel_l2"])
    print(f"  PASS in-place mutation + accumulate_dk: cos={m_acc.cosine_similarity:.6f}, rel_l2={m_acc.rel_l2_error:.6e}")

    # 9. Typical V4.1-Flash long-context / CP shape (Spec Section 6.2 #1)
    if include_large:
        B, S_q, S_k = 1, 32768, 16384
        q_l = torch.randn(B, S_q, H, D, dtype=torch.bfloat16)
        k_l = torch.randn(B, S_k, D, dtype=torch.bfloat16)
        w_l = torch.randn(B, S_q, H, dtype=torch.float32)
        dI_l = torch.randn(B, S_q, K, dtype=torch.float32)
        idx_l = make_fast_large_indices(B, S_q, S_k, K, sink_tokens=4, seed=99)
        _check_case_npu(
            "v41_flash_long_context_32k_16k", q_l, k_l, w_l, idx_l, dI_l,
            scale=1.0 / math.sqrt(D), use_analytical_ref=True, **common_kw,
        )

    print(
        f"\nALL NPU ACCURACY CHECKS PASSED (cores={num_cores}, BK={bk}, "
        f"staged={staged_pipeline}, interleaved={interleaved_pipeline})!"
    )

    if bench:
        from tilelang.profiler import do_bench

        B, S_q, S_k = 1, 32768, 16384
        q_b = torch.randn(B, S_q, H, D, dtype=torch.bfloat16).to("npu")
        k_b = torch.randn(B, S_k, D, dtype=torch.bfloat16).to("npu")
        w_b = torch.randn(B, S_q, H, dtype=torch.float32).to("npu")
        dI_b = torch.randn(B, S_q, K, dtype=torch.float32).to("npu")
        dq_b = torch.empty((B, S_q, H, D), dtype=torch.bfloat16, device="npu")
        dk_b = torch.empty((B, S_k, D), dtype=torch.bfloat16, device="npu")
        dw_b = torch.empty((B, S_q, H), dtype=torch.float32, device="npu")
        dk_ws_b = torch.empty((B, S_k, D), dtype=torch.float32, device="npu")

        total_flops = 6.0 * B * S_q * H * D * K
        peak_tflops = {28: 378.0, 32: 432.0, 36: 486.0}.get(num_cores, 432.0 * num_cores / 32.0)

        candidates = [
            ("bk128_v1_staged", dict(bk=128, staged_pipeline=True, interleaved_pipeline=False, dk_l2_policy="NORMAL_FV", k_l2_policy="NORMAL_LV")),
            ("bk128_v4_5stage", dict(bk=128, staged_pipeline=True, interleaved_pipeline=True, dk_l2_policy="NORMAL_FV", k_l2_policy="NORMAL_FV")),
            ("bk64_v4_5stage", dict(bk=64, staged_pipeline=True, interleaved_pipeline=True, dk_l2_policy="NORMAL_FV", k_l2_policy="NORMAL_FV")),
        ]

        patterns = [
            ("window_sink4", make_fast_large_indices(B, S_q, S_k, K, sink_tokens=4, seed=99).to("npu")),
            ("scattered_sink4", make_scattered_large_indices(B, S_q, S_k, K, sink_tokens=4).to("npu")),
        ]

        for pat_name, idx_b in patterns:
            print(
                f"\n=== NPU AB/BA Paired Benchmark [{pat_name}] "
                f"(cores={num_cores} AIC, B=1, S_q=32768, S_k=16384, H=32, D=128, K=512) ==="
            )
            fns = {}
            for cname, ccfg in candidates:
                def _make(c=ccfg, idx_t=idx_b):
                    return lambda: lightning_indexer_sparse_bwd(
                        q_b, k_b, w_b, idx_t, dI_b,
                        scale=1.0 / math.sqrt(D),
                        dk_workspace=dk_ws_b,
                        num_cores=num_cores,
                        dq_out=dq_b,
                        dk_out=dk_b,
                        dw_out=dw_b,
                        **c,
                    )
                fns[cname] = _make()
                fns[cname]()
            torch.npu.synchronize()

            samples = {cname: [] for cname, _ in candidates}
            for rnd in range(3):
                order = candidates if (rnd % 2 == 0) else list(reversed(candidates))
                for cname, _ in order:
                    prof = do_bench(
                        fns[cname],
                        backend="msprof_detail",
                        _n_warmup=5,
                        _n_repeat=15,
                        early_stop_baseline=None,
                    )
                    samples[cname].append(prof)

            for cname, _ in candidates:
                durs_ms = [p.dur_ns / 1e6 for p in samples[cname]]
                med_ms = statistics.median(durs_ms)
                tflops = total_flops / (med_ms * 1e9)
                mfu = 100.0 * tflops / peak_tflops
                p0 = samples[cname][-1]
                print(
                    f"  {cname:22s}: median={med_ms:.3f} ms (rounds={[round(x, 3) for x in durs_ms]}) | "
                    f"{tflops:.2f} TFLOPS | Cube MFU={mfu:.2f}% | "
                    f"AIC(MAD={p0.aic_mad:.1%}, FIX={p0.aic_fixpipe:.1%}) "
                    f"AIV(VEC={p0.aiv_vec:.1%}, MTE2={p0.aiv_mte2:.1%}, MTE3={p0.aiv_mte3:.1%})"
                )


def run_npu_profile_only(
    *,
    bk: int = 128,
    num_cores: int = 32,
    staged_pipeline: bool = True,
    interleaved_pipeline: bool = True,
    dk_l2_policy: str = "NORMAL_FV",
    k_l2_policy: str = "NORMAL_FV",
    sorted_invalid_tail: bool = True,
    batch_size: int = 1,
    seq_q: int = 32768,
    seq_k: int = 16384,
    sink_tokens: int = 4,
    index_pattern: str = "window",
    warmup: int = 0,
    repeat: int = 1,
    only_main_kernel: bool = False,
) -> None:
    """Launch a single operator configuration on NPU with zero accuracy verification.

    Designed for direct wrapping under ``msprof op`` so hardware pipe timelines
    and counters can be captured without CPU reference or multi-case overhead.
    """
    import torch_npu  # noqa: F401
    from examples.ascend.lightning_indexer.kernel import (
        _get_compiled_kernel,
        lightning_indexer_sparse_bwd,
    )

    assert torch.npu.is_available(), "torch.npu is not available in this environment."
    H, D, K = 32, 128, 512
    B, S_q, S_k = batch_size, seq_q, seq_k
    N_q, N_k = B * S_q, B * S_k
    scale = 1.0 / math.sqrt(D)

    # Build all tensors on CPU and copy via H2D so toolkit-only CANN installs work cleanly.
    torch.manual_seed(99)
    q = torch.randn(B, S_q, H, D, dtype=torch.bfloat16).to("npu")
    k = torch.randn(B, S_k, D, dtype=torch.bfloat16).to("npu")
    w = torch.randn(B, S_q, H, dtype=torch.float32).to("npu")
    dI = torch.randn(B, S_q, K, dtype=torch.float32).to("npu")
    if index_pattern == "scattered":
        idx = make_scattered_large_indices(B, S_q, S_k, K, sink_tokens=sink_tokens).to("npu")
    else:
        idx = make_fast_large_indices(B, S_q, S_k, K, sink_tokens=sink_tokens, seed=99).to("npu")

    dq = torch.empty((B, S_q, H, D), dtype=torch.bfloat16, device="npu")
    dk = torch.empty((B, S_k, D), dtype=torch.bfloat16, device="npu")
    dw = torch.empty((B, S_q, H), dtype=torch.float32, device="npu")
    dk_ws = torch.zeros((B, S_k, D), dtype=torch.float32).to("npu")

    scope_desc = "main-kernel-only" if only_main_kernel else "full-op (zero+main+cast)"
    print(
        f"=== NPU Profile-Only Launch ({scope_desc}) ===\n"
        f"  shape : B={B}, S_q={S_q}, S_k={S_k}, H={H}, D={D}, K={K}, "
        f"pattern={index_pattern}, sink_tokens={sink_tokens}\n"
        f"  config: num_cores={num_cores} AIC ({num_cores * 2} AIV), BK={bk}, "
        f"staged={staged_pipeline}, interleaved={interleaved_pipeline}, "
        f"k_l2={k_l2_policy}, dk_l2={dk_l2_policy}, sorted_tail={sorted_invalid_tail}\n"
        f"  iters : warmup={warmup}, repeat={repeat}"
    )

    if only_main_kernel:
        kernel = _get_compiled_kernel(
            H, D, K, bk, num_cores, sorted_invalid_tail, 2,
            staged_pipeline, interleaved_pipeline, dk_l2_policy, k_l2_policy,
        )
        q_flat = q.view(N_q, H, D)
        k_flat = k.view(N_k, D)
        w_flat = w.view(N_q, H)
        idx_flat = idx.view(N_q, K)
        dI_flat = dI.view(N_q, K)
        dq_flat = dq.view(N_q, H, D)
        dk_ws_flat = dk_ws.view(N_k, D)
        dw_flat = dw.view(N_q, H)

        def _launch() -> None:
            kernel(q_flat, k_flat, w_flat, idx_flat, dI_flat, dq_flat, dk_ws_flat, dw_flat, int(S_q), int(S_k), float(scale))
    else:
        def _launch() -> None:
            lightning_indexer_sparse_bwd(
                q, k, w, idx, dI,
                scale=scale,
                dk_workspace=dk_ws,
                sorted_invalid_tail=sorted_invalid_tail,
                bk=bk,
                num_cores=num_cores,
                staged_pipeline=staged_pipeline,
                interleaved_pipeline=interleaved_pipeline,
                dk_l2_policy=dk_l2_policy,
                k_l2_policy=k_l2_policy,
                dq_out=dq,
                dk_out=dk,
                dw_out=dw,
            )

    torch.npu.synchronize()
    for _ in range(warmup):
        _launch()
    if warmup > 0:
        torch.npu.synchronize()

    for _ in range(repeat):
        _launch()
    torch.npu.synchronize()
    print("  DONE: profile-only launch completed cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("cpu-ref", "compile", "npu", "profile"),
        default="cpu-ref",
        help=(
            "Execution mode: cpu-ref (no NPU), compile (CANN .o + CCE audit), "
            "npu (full Ascend 950 accuracy), profile (single-config NPU launch only, no accuracy check, for msprof op)"
        ),
    )
    parser.add_argument("--bk", type=int, choices=(64, 128), default=128, help="K-tile size for NPU/profile mode")
    parser.add_argument(
        "--num-cores",
        type=int,
        default=32,
        help="AI Core count (32=acceptance standard, 28=28-AIC SKU, 36=full 950DT)",
    )
    parser.add_argument(
        "--compile-cores",
        type=int,
        nargs="+",
        default=[28, 32, 36],
        help="Core counts to compile and audit in --mode compile",
    )
    parser.add_argument(
        "--staged-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable cross-iteration pipeline overlap",
    )
    parser.add_argument(
        "--interleaved-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable V2 cross-query flattened pipeline with interleaved MTE2||MTE3 + 8/4-row coalescing",
    )
    parser.add_argument("--dk-l2-policy", type=str, default="NORMAL_FV", help="L2 store policy for dK_fp32 atomic scatter")
    parser.add_argument("--k-l2-policy", type=str, default="NORMAL_FV", help="L2 load policy for K gather")
    parser.add_argument(
        "--sorted-invalid-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable sorted invalid-tail early exit (effective in --no-staged-pipeline mode)",
    )
    parser.add_argument("--skip-large", action="store_true", help="Skip 32K/16K case in NPU mode")
    parser.add_argument("--bench", action="store_true", help="Run AB/BA latency + msprof benchmark after NPU accuracy")
    parser.add_argument("--dump-source", type=str, default=None, help="Directory to save generated CCE sources in compile mode")
    # Profile-only options (for msprof op capture)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size B for --mode profile")
    parser.add_argument("--seq-q", type=int, default=32768, help="Query length S_q for --mode profile")
    parser.add_argument("--seq-k", type=int, default=16384, help="Key length S_k for --mode profile")
    parser.add_argument("--sink-tokens", type=int, default=4, help="Shared sink tokens per query for --mode profile")
    parser.add_argument(
        "--index-pattern",
        choices=("window", "scattered"),
        default="window",
        help="Index distribution for --mode profile: window (sliding window + sink) or scattered (0%% contiguous)",
    )
    parser.add_argument("--warmup", type=int, default=0, help="Warmup launches for --mode profile (default 0 for msprof op)")
    parser.add_argument("--repeat", type=int, default=1, help="Measured launches for --mode profile (default 1 for msprof op)")
    parser.add_argument(
        "--only-main-kernel",
        action="store_true",
        help="In --mode profile, launch ONLY the core Cube+Vector kernel (skip zero_fp32 and cast_fp32_to_bf16)",
    )
    args = parser.parse_args()

    if args.mode == "cpu-ref":
        run_cpu_reference_checks()
    elif args.mode == "compile":
        run_compile_checks(
            num_cores_list=tuple(args.compile_cores),
            staged_pipeline=args.staged_pipeline,
            interleaved_pipeline=args.interleaved_pipeline,
            dump_source_dir=args.dump_source,
        )
    elif args.mode == "npu":
        run_npu_verification(
            bk=args.bk,
            num_cores=args.num_cores,
            staged_pipeline=args.staged_pipeline,
            interleaved_pipeline=args.interleaved_pipeline,
            dk_l2_policy=args.dk_l2_policy,
            k_l2_policy=args.k_l2_policy,
            include_large=not args.skip_large,
            bench=args.bench,
        )
    elif args.mode == "profile":
        run_npu_profile_only(
            bk=args.bk,
            num_cores=args.num_cores,
            staged_pipeline=args.staged_pipeline,
            interleaved_pipeline=args.interleaved_pipeline,
            dk_l2_policy=args.dk_l2_policy,
            k_l2_policy=args.k_l2_policy,
            sorted_invalid_tail=args.sorted_invalid_tail,
            batch_size=args.batch_size,
            seq_q=args.seq_q,
            seq_k=args.seq_k,
            sink_tokens=args.sink_tokens,
            index_pattern=args.index_pattern,
            warmup=args.warmup,
            repeat=args.repeat,
            only_main_kernel=args.only_main_kernel,
        )


if __name__ == "__main__":
    main()
