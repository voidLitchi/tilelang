"""Run several QSA shapes through one dynamically compiled kernel."""

import argparse

import torch

from examples.ascend.qsa.compile import compile_qsa
from examples.ascend.qsa.kernel import HEAD_DIM, NUM_HEADS
from examples.ascend.qsa.testing import fill_contract_corners, make_inputs, ref_sparse_gqa, run_qsa


DEFAULT_SHAPES = ((16, 2048), (128, 65536), (4096, 32768))


def parse_shapes(value):
    shapes = []
    for item in value.split(","):
        try:
            num_tokens, kv_length = (int(part) for part in item.split(":", 1))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("shapes must use Q:KV,Q:KV notation") from exc
        if num_tokens < 1 or kv_length < 1:
            raise argparse.ArgumentTypeError("Q and KV must be positive")
        shapes.append((num_tokens, kv_length))
    if not shapes:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return tuple(shapes)


def check_rows(actual, q, k, v, indices, check_rows):
    if check_rows == 0:
        selected = torch.arange(q.shape[0], dtype=torch.long)
    else:
        selected = torch.linspace(
            0, q.shape[0] - 1, min(check_rows, q.shape[0]), dtype=torch.float64
        ).round().long()
    selected = selected.to(q.device)
    expected = ref_sparse_gqa(
        q.index_select(0, selected), k, v, indices.index_select(0, selected)
    )
    torch.testing.assert_close(
        actual.index_select(0, selected).float(), expected, atol=0.01, rtol=0.01
    )
    return selected.numel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=DEFAULT_SHAPES,
        help="comma-separated packed query/KV lengths, for example 16:2048,128:65536,4096:32768",
    )
    parser.add_argument("--top-k", type=int, default=2176)
    parser.add_argument("--variant", choices=("baseline", "old-schedule-tail"), default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-rows", type=int, default=8, help="0 checks every row")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.top_k <= 0 or args.top_k % 128:
        parser.error("top-k must be a positive multiple of 128")
    if args.check_rows < 0:
        parser.error("check-rows must be non-negative")

    first_q, first_kv = args.shapes[0]
    pad_m = 32 if args.variant == "baseline" else 16
    kernel = compile_qsa(
        first_q,
        first_kv,
        args.top_k,
        dynamic_shape=True,
        variant=args.variant,
        pad_m=pad_m,
        num_stages=3 if args.variant != "baseline" else None,
        aic_count=28 if args.variant != "baseline" else None,
    )
    print(
        f"compiled_once=True dynamic=True variant={args.variant} top_k={args.top_k} "
        f"head_dim={HEAD_DIM} q_heads={NUM_HEADS}",
        flush=True,
    )

    for index, (num_tokens, kv_length) in enumerate(args.shapes):
        q, k, v, indices = make_inputs(
            num_tokens, kv_length, args.top_k, seed=args.seed + index
        )
        fill_contract_corners(indices, kv_length, args.top_k)
        if args.variant == "baseline":
            actual = run_qsa(kernel, q, k, v, indices)
        else:
            safe = indices.clamp_min(0).to(torch.int32).contiguous()
            valid = (indices >= 0).to(torch.int32).contiguous()
            q_padded = q.reshape(num_tokens, 2, 12, 256)
            q_padded = torch.cat((q_padded, q_padded.new_zeros(num_tokens, 2, 4, 256)), dim=2)
            k_padded = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, 3))
            v_padded = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 3))
            output = kernel(q_padded.contiguous(), k_padded, v_padded, safe, valid)
            actual = output[:, :, :12, :].reshape(num_tokens, NUM_HEADS, HEAD_DIM)
        torch.npu.synchronize()

        checked = 0
        if args.check:
            checked = check_rows(actual, q, k, v, indices, args.check_rows)
        print(
            f"shape={num_tokens}:{kv_length} output={tuple(actual.shape)} "
            f"checked={checked} finite={bool(torch.isfinite(actual.float()).all())}",
            flush=True,
        )


if __name__ == "__main__":
    main()
