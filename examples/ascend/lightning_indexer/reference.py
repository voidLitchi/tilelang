"""PyTorch FP32/FP64 reference implementations and accuracy metrics for Lightning Indexer Sparse Backward.

Implements the v1 mathematical contract for DeepSeek-V4.1-Flash CSA2 / DSA sparse
Lightning Indexer backward propagation:

For each batch ``b``, query position ``t in [0, S_q)``, candidate slot ``m in [0, K)``,
and key index ``s = indices[b, t, m]``:

- Invalid slots (``s < 0`` or ``s >= S_k``) are masked out (no read, zero contribution).
- Forward:
    A[b, t, h, m] = scale * dot(q[b, t, h, :], k[b, s, :])
    R[b, t, h, m] = max(0, A[b, t, h, m])
    I[b, t, m]    = sum_h w[b, t, h] * R[b, t, h, m]   (0 if invalid)
- Backward:
    dA[b, t, h, m] = dI[b, t, m] * w[b, t, h] * 1(A[b, t, h, m] > 0) * 1(valid)
    dw[b, t, h]    = sum_{m valid} dI[b, t, m] * R[b, t, h, m]
    dq[b, t, h, :] = scale * sum_{m valid} dA[b, t, h, m] * k[b, s, :]
    dk[b, s, :]   += scale * sum_h dA[b, t, h, m] * q[b, t, h, :]

Both a chunked ``torch.autograd.grad`` reference (gold standard, independent of
backward derivation) and an independent closed-form analytical FP32 reference are
provided so they can cross-check each other and scale to long-context shapes
such as ``(B=1, S_q=32768, S_k=16384, H=32, D=128, K=512)`` without OOM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def ref_lightning_indexer_sparse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    indices: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Differentiable PyTorch reference for sparse Lightning Indexer forward.

    Preserves the floating-point dtype of ``q``/``k``/``w`` so callers can run
    FP32 reference evaluations or FP64 ``torch.autograd.gradcheck``.

    Args:
        q: ``[B, S_q, H, D]`` floating-point tensor.
        k: ``[B, S_k, D]`` floating-point tensor.
        w: ``[B, S_q, H]`` floating-point tensor.
        indices: ``[B, S_q, K]`` integer tensor with values in ``[0, S_k)`` or invalid.
        scale: Dot-product scaling factor ``alpha``.

    Returns:
        ``I`` of shape ``[B, S_q, K]`` with invalid slots strictly zeroed.
    """
    B, S_q, H, D = q.shape
    B_k, S_k, D_k = k.shape
    assert B == B_k and D == D_k, f"Shape mismatch: q={q.shape}, k={k.shape}"
    assert w.shape == (B, S_q, H), f"Shape mismatch: w={w.shape}, expected {(B, S_q, H)}"
    assert indices.shape[:2] == (B, S_q), f"Shape mismatch: indices={indices.shape}"
    assert S_k > 0, "S_k must be positive"

    valid = (indices >= 0) & (indices < S_k)  # [B, S_q, K]
    safe_idx = indices.clamp(0, S_k - 1).to(torch.long)  # [B, S_q, K]

    batch_idx = torch.arange(B, device=k.device, dtype=torch.long).view(B, 1, 1)
    k_gathered = k[batch_idx, safe_idx]  # [B, S_q, K, D]
    k_gathered = k_gathered * valid.unsqueeze(-1).to(k_gathered.dtype)

    # A: [B, S_q, H, K]
    A = torch.einsum("bqhd,bqkd->bqhk", q, k_gathered) * scale
    R = F.relu(A)
    I = torch.einsum("bqh,bqhk->bqk", w, R)
    I = torch.where(valid, I, torch.zeros_like(I))
    return I


def ref_lightning_indexer_sparse_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    indices: torch.Tensor,
    dI: torch.Tensor,
    scale: float = 1.0,
    q_chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gold-standard FP32 backward reference via chunked ``torch.autograd.grad``.

    Chunking along ``S_q`` keeps temporary memory bounded (under ~100 MB even
    for ``S_q=32768, H=32, D=128, K=512``) while deriving gradients directly
    from the forward mathematical definition.

    Returns:
        ``(dq_fp32, dk_fp32, dw_fp32)`` all in ``torch.float32``.
    """
    B, S_q, H, D = q.shape
    _, S_k, _ = k.shape
    device = q.device

    q_fp32 = q.detach().to(torch.float32)
    k_fp32 = k.detach().to(torch.float32)
    w_fp32 = w.detach().to(torch.float32)
    dI_fp32 = dI.detach().to(torch.float32)

    dq_out = torch.empty((B, S_q, H, D), dtype=torch.float32, device=device)
    dk_out = torch.zeros((B, S_k, D), dtype=torch.float32, device=device)
    dw_out = torch.empty((B, S_q, H), dtype=torch.float32, device=device)

    with torch.enable_grad():
        for t0 in range(0, S_q, q_chunk_size):
            t1 = min(S_q, t0 + q_chunk_size)
            q_c = q_fp32[:, t0:t1].detach().requires_grad_(True)
            k_c = k_fp32.detach().requires_grad_(True)
            w_c = w_fp32[:, t0:t1].detach().requires_grad_(True)
            idx_c = indices[:, t0:t1]
            dI_c = dI_fp32[:, t0:t1]

            I_c = ref_lightning_indexer_sparse_fwd(q_c, k_c, w_c, idx_c, scale=scale)
            dq_c, dk_c, dw_c = torch.autograd.grad(
                outputs=I_c,
                inputs=(q_c, k_c, w_c),
                grad_outputs=dI_c,
            )
            dq_out[:, t0:t1] = dq_c
            dk_out += dk_c
            dw_out[:, t0:t1] = dw_c

    return dq_out, dk_out, dw_out


@torch.no_grad()
def ref_lightning_indexer_sparse_bwd_analytical(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    indices: torch.Tensor,
    dI: torch.Tensor,
    scale: float = 1.0,
    q_chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent closed-form FP32 analytical backward reference.

    Matches Section 3 of ``lightning_indexer_bwd_spec.md`` directly without
    building an autograd graph; useful both to cross-check the autograd
    reference and for fast reference evaluation on large shapes.

    Returns:
        ``(dq_fp32, dk_fp32, dw_fp32)`` all in ``torch.float32``.
    """
    B, S_q, H, D = q.shape
    _, S_k, _ = k.shape
    device = q.device

    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    w_fp32 = w.to(torch.float32)
    dI_fp32 = dI.to(torch.float32)

    dq_out = torch.empty((B, S_q, H, D), dtype=torch.float32, device=device)
    dk_out = torch.zeros((B, S_k, D), dtype=torch.float32, device=device)
    dw_out = torch.empty((B, S_q, H), dtype=torch.float32, device=device)
    batch_idx = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1)

    for t0 in range(0, S_q, q_chunk_size):
        t1 = min(S_q, t0 + q_chunk_size)
        q_c = q_fp32[:, t0:t1]  # [B, C, H, D]
        w_c = w_fp32[:, t0:t1]  # [B, C, H]
        idx_c = indices[:, t0:t1]  # [B, C, K]
        dI_c = dI_fp32[:, t0:t1]  # [B, C, K]

        valid = (idx_c >= 0) & (idx_c < S_k)  # [B, C, K]
        safe_idx = idx_c.clamp(0, S_k - 1).to(torch.long)  # [B, C, K]
        k_g = k_fp32[batch_idx, safe_idx]  # [B, C, K, D]
        k_g = k_g * valid.unsqueeze(-1).to(torch.float32)

        # Forward recomputation: A = scale * Q @ K^T
        A = torch.einsum("bqhd,bqkd->bqhk", q_c, k_g) * scale
        valid_pos = valid.unsqueeze(2) & (A > 0)
        R = torch.where(valid_pos, A, torch.zeros_like(A))

        # dw = sum_{m valid} dI * R
        dw_c = (R * dI_c.unsqueeze(2)).sum(dim=-1)
        dw_out[:, t0:t1] = dw_c

        # dA = 1(A > 0 & valid) * w * dI
        dA = torch.where(
            valid_pos,
            w_c.unsqueeze(-1) * dI_c.unsqueeze(2),
            torch.zeros_like(A),
        )

        # dq = scale * dA @ K
        dq_out[:, t0:t1] = torch.einsum("bqhk,bqkd->bqhd", dA, k_g) * scale

        # dk_local = scale * dA^T @ Q -> scatter-add to dk_out
        dk_local = torch.einsum("bqhk,bqhd->bqkd", dA, q_c) * scale
        dk_local = dk_local * valid.unsqueeze(-1).to(torch.float32)

        for b in range(B):
            valid_b = valid[b].reshape(-1)
            if valid_b.any():
                idx_b = safe_idx[b].reshape(-1)[valid_b]
                src_b = dk_local[b].reshape(-1, D)[valid_b]
                dk_out[b].index_add_(0, idx_b, src_b)

    return dq_out, dk_out, dw_out


@dataclass(frozen=True)
class AccuracyMetrics:
    """Numerical comparison summary between candidate and FP32 reference."""

    name: str
    dtype: str
    has_nan: bool
    has_inf: bool
    cosine_similarity: float
    rel_l2_error: float
    p999_abs_error: float
    max_abs_error: float
    ref_norm: float
    actual_norm: float

    def passes(self, *, min_cos: float, max_rel_l2: float, zero_atol: float = 1e-6) -> bool:
        if self.has_nan or self.has_inf:
            return False
        if self.ref_norm == 0.0:
            return self.max_abs_error <= zero_atol
        return self.cosine_similarity >= min_cos and self.rel_l2_error <= max_rel_l2


@torch.no_grad()
def compute_accuracy_metrics(name: str, actual: torch.Tensor, expected_fp32: torch.Tensor) -> AccuracyMetrics:
    """Compute Spec Section 6.1 accuracy metrics against FP32 reference."""
    act_f = actual.detach().to(torch.float32).reshape(-1)
    exp_f = expected_fp32.detach().to(torch.float32).reshape(-1)
    assert act_f.shape == exp_f.shape, f"{name} shape mismatch: {actual.shape} vs {expected_fp32.shape}"

    has_nan = bool(torch.isnan(act_f).any().item())
    has_inf = bool(torch.isinf(act_f).any().item())
    if has_nan or has_inf:
        return AccuracyMetrics(
            name=name,
            dtype=str(actual.dtype),
            has_nan=has_nan,
            has_inf=has_inf,
            cosine_similarity=float("nan"),
            rel_l2_error=float("inf"),
            p999_abs_error=float("inf"),
            max_abs_error=float("inf"),
            ref_norm=float(exp_f.norm().item()),
            actual_norm=float("nan"),
        )

    diff = act_f - exp_f
    abs_diff = diff.abs()
    ref_norm = float(exp_f.norm().item())
    act_norm = float(act_f.norm().item())
    diff_norm = float(diff.norm().item())
    max_abs = float(abs_diff.max().item()) if abs_diff.numel() > 0 else 0.0

    # Sample-free exact 99.9% quantile on CPU/small tensors; kthvalue works on large 1D tensors too.
    if abs_diff.numel() == 0:
        p999_abs = 0.0
    else:
        kth = max(1, min(abs_diff.numel(), int(0.999 * abs_diff.numel())))
        p999_abs = float(abs_diff.kthvalue(kth).values.item())

    if ref_norm == 0.0:
        cos_sim = 1.0 if act_norm == 0.0 else 0.0
        rel_l2 = 0.0 if act_norm == 0.0 else float("inf")
    elif act_norm == 0.0:
        cos_sim = 0.0
        rel_l2 = 1.0
    else:
        cos_sim = float((torch.dot(act_f, exp_f) / (act_norm * ref_norm)).clamp(-1.0, 1.0).item())
        rel_l2 = diff_norm / ref_norm

    return AccuracyMetrics(
        name=name,
        dtype=str(actual.dtype),
        has_nan=False,
        has_inf=False,
        cosine_similarity=cos_sim,
        rel_l2_error=rel_l2,
        p999_abs_error=p999_abs,
        max_abs_error=max_abs,
        ref_norm=ref_norm,
        actual_norm=act_norm,
    )


# Spec Section 6.1 acceptance thresholds
SPEC_THRESHOLDS = {
    "dq": {"min_cos": 0.9995, "max_rel_l2": 1.0e-2},
    "dk": {"min_cos": 0.9995, "max_rel_l2": 1.0e-2},
    "dw": {"min_cos": 0.9998, "max_rel_l2": 5.0e-3},
}
