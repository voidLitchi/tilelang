"""Compare QSA and CANN GBSA using one set of inputs and causal selections."""

import argparse
import random
import time

import torch


def block_lists(batch, queries, kv, seed):
    rng = random.Random(seed)
    lists = torch.full((batch * queries, 512), -1, dtype=torch.int32)
    counts = torch.empty(batch * queries, dtype=torch.int32)
    for row in range(batch * queries):
        complete = (kv - queries + row % queries + 1) // 4
        count = min(complete, 512)
        counts[row] = count
        lists[row, :count] = torch.tensor(sorted(rng.sample(range(complete), count)), dtype=torch.int32)
    return lists, counts


def token_slots(lists, counts, queries, kv):
    rows = torch.arange(len(counts), device=lists.device, dtype=torch.int32)
    visible = kv - queries + rows % queries + 1
    offsets = (rows // queries * kv)[:, None]
    lane = torch.arange(4, device=lists.device, dtype=torch.int32)
    full = (lists[:, :, None] * 4 + lane).flatten(1)
    slot = torch.arange(2176, device=lists.device, dtype=torch.int32)[None, :]
    end = 4 * counts[:, None]
    tail = visible[:, None] % 4
    full = torch.nn.functional.pad(full, (0, 128))
    return torch.where(slot < end, offsets + full, torch.where(slot < end + tail, offsets + visible[:, None] - tail + slot - end, -1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--kv", type=int, default=2055)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-rows", type=int, default=8, help="0 checks all rows")
    parser.add_argument("--scope", choices=("prepared", "adapter", "both"), default="both")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--pv-ready", action="store_true", help="enable the QSA PV synchronization diagnostic when compiling TileLang")
    parser.add_argument("--probe-only", action="store_true", help="check the installed GBSA metadata contract without compiling QSA")
    args = parser.parse_args()
    batch, queries, kv = args.batch, args.queries, args.kv
    if min(batch, queries, args.repeat) < 1 or kv < queries or args.check_rows < 0 or batch * kv >= 2**31:
        parser.error("require batch, queries, repeat > 0; kv >= queries; check-rows >= 0; batch*kv < 2^31")

    import cann_ops_transformer
    import torch_npu

    device = torch.device("npu", torch.npu.current_device())
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    total = batch * queries
    cpu_lists, cpu_counts = block_lists(batch, queries, kv, args.seed)
    lists, counts = cpu_lists.to(device), cpu_counts.to(device)
    common = dict(
        layout_q="TND",
        layout_kv="TND",
        mask_mode=1,
        quant_mode=0,
        softmax_precision=0,
        residual_block_mode=1,
        cu_seqlens_q=torch.arange(batch + 1, device=device, dtype=torch.int64) * queries,
        cu_seqlens_kv=torch.arange(batch + 1, device=device, dtype=torch.int64) * kv,
    )
    ops = cann_ops_transformer.ops

    def tl_inputs():
        return pad_q_heads(query), *split_indices(token_slots(lists, counts, queries, kv))

    def gb_inputs():
        blocks = lists.unsqueeze(0).expand(2, -1, -1).contiguous()
        sizes = counts.unsqueeze(0).expand(2, -1).contiguous()
        metadata = ops.generic_block_sparse_attention_metadata(
            blocks, sizes, 24, 2, 256, [1, 4], max_seqlen_q=queries, max_seqlen_kv=kv, **common
        )
        return blocks, sizes, metadata

    def tl_run(padded, indices, valid):
        return kernel(padded, key, value, indices, valid)

    def gb_run(blocks, sizes, metadata):
        return ops.generic_block_sparse_attention(
            query, key, value, blocks, sizes, [1, 4], metadata=metadata, softmax_scale=1 / 16, **common
        )[0]

    def unpad(output):
        return output[:, :, :12].reshape(total, 24, 256)

    try:
        gbsa_inputs = gb_inputs()
        torch.npu.synchronize()
    except RuntimeError as exc:
        if "kvInputLayout currently only supports PA_BBND" not in str(exc):
            raise
        raise RuntimeError(
            "Installed GBSA metadata rejects TND KV. On Ascend 950 the stock v9.3.0-beta.1 operator also "
            "requires D=128, blockShape=[1,128], topK<=256 and residualBlockMode=0. Changing layout alone "
            "cannot benchmark QSA (D=256, [1,4], 512 blocks + tail). Install matching main and metadata "
            "operators and torch extension with QSA support; see COMPARE.md. No performance ratio computed."
        ) from exc
    if args.probe_only:
        print("GBSA metadata accepted QSA TND parameters; attention execution and correctness were not tested.")
        return

    import tilelang
    from tilelang.profiler import do_bench
    from examples.ascend.qsa.compile import compile_qsa
    from examples.ascend.qsa.testing import pad_q_heads, split_indices

    if args.pv_ready:
        from examples.ascend.qsa.diagnose_pv_sync import enable_pv_ready

        enable_pv_ready()

    query = torch.randn(total, 24, 256, dtype=torch.bfloat16, device=device)
    key = torch.randn(batch * kv, 2, 256, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    kernel = compile_qsa(total, batch * kv, 2176, dynamic_shape=args.dynamic)
    tile_inputs = tl_inputs()
    print(
        f"device={torch.npu.get_device_name(device.index)} torch={torch.__version__} torch_npu={torch_npu.__version__} "
        f"tilelang={tilelang.__file__} batch={batch} q/request={queries} kv/request={kv} slots=2176 seed={args.seed} dynamic={args.dynamic}"
    )
    if args.check:
        tile_output, gb_output = unpad(tl_run(*tile_inputs)), gb_run(*gbsa_inputs)
        checked = min(args.check_rows or total, total)
        selected = torch.linspace(0, total - 1, checked, dtype=torch.float64).round().long().tolist()
        for row in selected:
            visible = kv - queries + row % queries + 1
            blocks = cpu_lists[row, : int(cpu_counts[row])].long()
            tokens = torch.cat(((4 * blocks[:, None] + torch.arange(4)).flatten(), torch.arange(visible // 4 * 4, visible)))
            ids = (tokens + row // queries * kv).to(device)
            logits = torch.einsum("hgd,nhd->hgn", query[row].cpu().float().view(2, 12, 256), key.index_select(0, ids).cpu().float()) / 16
            oracle = torch.einsum("hgn,nhd->hgd", logits.softmax(-1), value.index_select(0, ids).cpu().float()).reshape(24, 256)
            for label, output in (("TileLang", tile_output), ("GBSA", gb_output)):
                torch.testing.assert_close(
                    output[row].cpu().float(),
                    oracle,
                    atol=0.01,
                    rtol=0.01,
                    msg=lambda msg, name=label, index=row: f"{name} row={index}: {msg}",
                )
        print(f"CPU FP32 verification passed: {checked}/{total} rows")
    for scope, tile_fn, gb_fn in (
        ("prepared", lambda: tl_run(*tile_inputs), lambda: gb_run(*gbsa_inputs)),
        ("adapter", lambda: unpad(tl_run(*tl_inputs())), lambda: gb_run(*gb_inputs())),
    ):
        if args.scope not in (scope, "both"):
            continue

        def measure(fn, timing_scope):
            if timing_scope == "adapter":
                for _ in range(10):
                    fn()
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(args.repeat):
                    fn()
                torch.npu.synchronize()
                return (time.perf_counter() - start) * 1e6 / args.repeat
            return 1000 * do_bench(fn, backend="msprof", device=device, _n_warmup=10, _n_repeat=args.repeat)

        tile_us, gb_us = measure(tile_fn, scope), measure(gb_fn, scope)
        ratio = gb_us / tile_us
        print(
            f"{scope}: TileLang={tile_us:.2f} us  GBSA={gb_us:.2f} us  GBSA_us/TileLang_us={ratio:.3f}  "
            f"{'PASS' if ratio >= 0.9 else 'BELOW'} 0.90x"
        )


if __name__ == "__main__":
    main()
