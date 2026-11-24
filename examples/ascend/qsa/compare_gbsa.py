"""Measure the PR's dynamic QSA and ops-transformer GBSA on identical packed inputs."""

import argparse
import csv
import math
import random
import statistics
import sys

import torch

from examples.ascend.qsa.compile import compile_qsa
from examples.ascend.qsa.run import prepare_inputs
from tilelang.profiler import do_bench


# Request tuples contain (number of new Q tokens, KV length including the new tokens).
CASES = {
    "C01": [(1, 24576)],
    "C02": [(4, 24579)],
    "C03": [(1, 24577), (14, 32770), (15, 65539)],
    "C04": [(1, 32768)] * 32,
    "C05": [(1, 24577)] * 128,
    "C06": [(1, 65536)],
    "C07": [(1, 262144)],
    "C08": [(1, 1048576)],
    "C09": [(4096, 32768)],
    "C10": [(4096, 262144)],
    "C11": [(4096, 65536)] * 4,
    "C12": [(32768, 262144)],
    "C13": [(4096, 1048576)],
}
FIELDS = (
    "case",
    "batch",
    "T",
    "S_kv",
    "rows",
    "qsa_max_abs",
    "gbsa_max_abs",
    "qsa_bad",
    "gbsa_bad",
    "qsa_us",
    "gbsa_us",
    "speedup",
    "performance",
    "status",
    "detail",
)


def selections(requests, seed):
    rng = random.Random(seed)
    total = sum(q for q, _ in requests)
    blocks = torch.full((total, 512), -1, dtype=torch.int32)
    counts = torch.empty(total, dtype=torch.int32)
    slots = torch.full((total, 2176), -1, dtype=torch.int32)
    row = start = 0
    row_info = []
    lane = torch.arange(4, dtype=torch.int32)
    for q, kv in requests:
        for visible in range(kv - q + 1, kv + 1):
            n, tail = divmod(visible, 4)
            count = min(512, n)
            selected = torch.tensor(sorted(rng.sample(range(n), count)), dtype=torch.int32)
            blocks[row, :count] = selected
            counts[row] = count
            slots[row, : 4 * count] = (start + 4 * selected[:, None] + lane).flatten()
            slots[row, 4 * count : 4 * count + tail] = start + 4 * n + lane[:tail]
            row_info.append((start, visible))
            row += 1
        start += kv
    return blocks, counts, slots, row_info


def gbsa_runner(ops, q, k, v, requests, blocks, counts):
    q_lens, kv_lens = zip(*requests, strict=True)
    device = q.device
    common = dict(
        layout_q="TND",
        layout_kv="TND",
        mask_mode=1,
        quant_mode=0,
        softmax_precision=0,
        residual_block_mode=1,
        cu_seqlens_q=torch.tensor((0, *q_lens), device=device, dtype=torch.int64).cumsum(0),
        cu_seqlens_kv=torch.tensor((0, *kv_lens), device=device, dtype=torch.int64).cumsum(0),
    )
    lists = blocks.to(device).unsqueeze(0).expand(2, -1, -1).contiguous()
    sizes = counts.to(device).unsqueeze(0).expand(2, -1).contiguous()
    metadata = ops.generic_block_sparse_attention_metadata(
        lists, sizes, 24, 2, 256, [1, 4], max_seqlen_q=max(q_lens), max_seqlen_kv=max(kv_lens), **common
    )
    return lambda: ops.generic_block_sparse_attention(q, k, v, lists, sizes, [1, 4], metadata=metadata, softmax_scale=1 / 16, **common)[0]


def check_outputs(outputs, q, k, v, blocks, counts, row_info, rows):
    # Reconstruct logical tokens independently from the GBSA block list and
    # the causal boundary, without consulting QSA's padded token slot tensor.
    errors = {name: [0.0, 0] for name in outputs}
    for row in rows:
        start, visible = row_info[row]
        selected = blocks[row, : int(counts[row])].long()
        ids = torch.cat(((4 * selected[:, None] + torch.arange(4)).flatten(), torch.arange(visible // 4 * 4, visible)))
        ids = (start + ids).to(k.device)
        query = q[row].cpu().float().reshape(2, 12, 256)
        key = k.index_select(0, ids).cpu().float()
        value = v.index_select(0, ids).cpu().float()
        scores = torch.einsum("hgd,nhd->hgn", query, key) * 0.0625
        expected = torch.einsum("hgn,nhd->hgd", scores.softmax(-1), value).reshape(24, 256)
        for name, output in outputs.items():
            err = (output[row].cpu().float() - expected).abs()
            worst = err.max().item()
            errors[name][0] = max(errors[name][0], worst) if math.isfinite(worst) else math.inf
            errors[name][1] += int((~(err <= 0.01 + 0.01 * expected.abs())).sum())
    return errors


def run_case(result, kernel, ops, requests, args):
    total = result["T"]
    device = torch.device("npu", torch.npu.current_device())
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    blocks, counts, slots, info = selections(requests, args.seed)
    q = torch.randn(total, 24, 256, dtype=torch.bfloat16, device=device)
    k = torch.randn(result["S_kv"], 2, 256, dtype=q.dtype, device=device)
    v = torch.randn_like(k)
    prepared = prepare_inputs(q, k, v, slots.to(device))
    runners = {"qsa": lambda: kernel(*prepared)}
    output = runners["qsa"]()
    torch.npu.synchronize()
    outputs = {"qsa": output[:, :, :12].reshape(total, 24, 256)}
    try:
        gb = gbsa_runner(ops, q, k, v, requests, blocks, counts)
        outputs["gbsa"] = gb()
        torch.npu.synchronize()
        runners["gbsa"] = gb
    except RuntimeError as exc:
        result.update(status="GBSA_ERROR", detail=str(exc))
        if "gbsa" in outputs:
            del outputs["gbsa"]

    count = total if args.check_rows == 0 or (args.check_rows is None and total <= 128) else min(args.check_rows or 64, total)
    rows = torch.linspace(0, total - 1, count, dtype=torch.float64).round().long().tolist()
    result["rows"] = count
    errors = check_outputs(outputs, q, k, v, blocks, counts, info, rows)
    for name, (worst, bad) in errors.items():
        result[name + "_max_abs"] = worst
        result[name + "_bad"] = bad
    if any(bad for _, bad in errors.values()):
        result["status"] = "PRECISION_FAIL"
        return

    samples = {name: [] for name in runners}
    for i in range(args.rounds):
        order = list(runners) if i % 2 == 0 else list(reversed(runners))
        for name in order:
            us = 1000 * do_bench(runners[name], backend="msprof", device=device, _n_warmup=10, _n_repeat=args.repeat)
            if not math.isfinite(us) or us <= 0:
                raise ValueError(f"invalid {name} latency {us}")
            samples[name].append(us)
    for name, times in samples.items():
        result[name + "_us"] = statistics.median(times)
    if len(samples) == 2:
        ratio = result["gbsa_us"] / result["qsa_us"]
        result.update(speedup=ratio, performance="PASS" if ratio >= 0.9 else "BELOW", status="OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    shapes = parser.add_mutually_exclusive_group()
    shapes.add_argument("--cases", nargs="+", choices=CASES)
    shapes.add_argument("--shape", nargs=3, type=int, metavar=("B", "Q", "KV"), help="one uniform packed batch")
    parser.add_argument("--aic-count", type=int, choices=(28, 32, 36), default=28)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check-rows", type=int, help="0 checks every row; default checks all up to 128 Q rows, otherwise 64")
    parser.add_argument("--csv", default="qsa-gbsa.csv")
    args = parser.parse_args()
    if min(args.rounds, args.repeat) < 1 or (args.check_rows is not None and args.check_rows < 0):
        parser.error("rounds and repeat must be positive and check-rows must be nonnegative")
    cases = {name: CASES[name] for name in args.cases or CASES}
    if args.shape:
        batch, q, kv = args.shape
        if batch < 1 or not 1 <= q <= kv or not 4 <= batch * kv <= 4194301 or batch * q > 262144:
            parser.error("require B >= 1, 1 <= Q <= KV, B*Q <= 262144 and 4 <= B*KV <= 4194301")
        cases = {"custom": [(q, kv)] * batch}
    import torch_npu
    import cann_ops_transformer

    print(
        f"{torch.npu.get_device_name(0)} torch={torch.__version__} torch_npu={torch_npu.__version__} "
        f"extension={cann_ops_transformer.__file__} aic={args.aic_count} seed={args.seed} rounds={args.rounds} repeat={args.repeat}",
        flush=True,
    )
    kernel = compile_qsa(1, 4, aic_count=args.aic_count, dynamic_shape=True)
    print("compiled_once=True", flush=True)
    failed = False
    with open(args.csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for case, requests in cases.items():
            result = dict(case=case, batch=len(requests), T=sum(q for q, _ in requests), S_kv=sum(k for _, k in requests))
            try:
                run_case(result, kernel, cann_ops_transformer.ops, requests, args)
            except (RuntimeError, ValueError, AssertionError) as exc:
                result.update(status="ERROR", detail=str(exc))
            finally:
                torch.npu.empty_cache()
            writer.writerow(result)
            file.flush()
            print(result, flush=True)
            failed |= result["status"] != "OK" or result.get("performance") == "BELOW"
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
