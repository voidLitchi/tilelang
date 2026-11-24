"""Compile one dynamic QSA kernel and reuse it across several shapes."""

import argparse

import torch

from examples.ascend.qsa.compile import compile_qsa
from examples.ascend.qsa.kernel import (
    HEAD_DIM,
    NUM_HEADS,
    SUPPORTED_AIC_COUNTS,
)
from examples.ascend.qsa.testing import fill_contract_corners, make_inputs, ref_sparse_gqa, run_qsa


DEFAULT_SHAPES = ((16, 2048), (128, 65536), (4096, 32768))


def parse_shapes(value):
    shapes = []
    for item in value.split(","):
        try:
            num_tokens, kv_length = (int(part) for part in item.split(":", 1))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("shapes must use Q:KV,Q:KV notation") from exc
        if num_tokens < 1 or kv_length < 4:
            raise argparse.ArgumentTypeError("Q and KV must be positive")
        shapes.append((num_tokens, kv_length))
    if not shapes:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return tuple(shapes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=DEFAULT_SHAPES,
        help="packed query/KV lengths, for example 16:2048,128:65536,4096:32768",
    )
    parser.add_argument("--aic-count", type=int, choices=SUPPORTED_AIC_COUNTS, default=32)
    parser.add_argument("--check-rows", type=int, default=8, help="0 checks every row")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    assert args.check_rows >= 0, "--check-rows must be non-negative"

    first_q, first_kv = args.shapes[0]
    kernel = compile_qsa(
        first_q,
        first_kv,
        dynamic_shape=True,
        aic_count=args.aic_count,
    )
    print(
        f"compiled_once=True dynamic=True aic_count={args.aic_count} q_heads={NUM_HEADS} head_dim={HEAD_DIM}",
        flush=True,
    )

    for index, (num_tokens, kv_length) in enumerate(args.shapes):
        q, k, v, indices = make_inputs(num_tokens, kv_length, seed=index)
        fill_contract_corners(indices, kv_length)
        actual = run_qsa(kernel, q, k, v, indices)
        torch.npu.synchronize()
        if args.check:
            expected = ref_sparse_gqa(q, k, v, indices)
            if args.check_rows == 0:
                selected = torch.arange(num_tokens, device=q.device)
            else:
                selected = (
                    torch.linspace(
                        0,
                        num_tokens - 1,
                        min(args.check_rows, num_tokens),
                        dtype=torch.float64,
                        device=q.device,
                    )
                    .round()
                    .long()
                )
            torch.testing.assert_close(
                actual.index_select(0, selected).float(),
                expected.index_select(0, selected),
                rtol=1e-2,
                atol=1e-2,
            )
        print(f"shape={num_tokens}:{kv_length} checked={args.check}", flush=True)


if __name__ == "__main__":
    main()
