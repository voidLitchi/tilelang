# QSA and GBSA on Ascend 950

Use the TileLang build for this checkout and a matching ops-transformer GBSA,
GBSA Metadata and `cann_ops_transformer` extension with the 950 BF16 TND/TND,
D=256, 1-by-4 block and residual-tail paths. The comparison targets the API at
ops-transformer commit `6945d02b7`. The stock CANN 9.3.0-beta.1 Metadata
operator that only accepts PA_BBND cannot run this comparison.

From the repository root with CANN and the custom OPP environment active:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD"
python3 -m examples.ascend.qsa.compare_gbsa --cases C02 C05 --csv smoke.csv
python3 -m examples.ascend.qsa.compare_gbsa --csv qsa-gbsa.csv
python3 -m examples.ascend.qsa.compare_gbsa --shape 1 4096 32768 --check-rows 0 --csv prefill.csv
```

The default suite covers C01 through C11, including a heterogeneous batch,
128-way decode, initial prefill and right-aligned prefill chunks up to a
262144-token context. Each invocation compiles one dynamic kernel for 28
AICs. Set `--aic-count 32` or `36` to match another device. The two kernels
share BF16 inputs and causal block selections. The CPU FP32 oracle reconstructs
the selected tokens independently of the prepared QSA slots. The default check
covers every query for T up to 128 and 64 evenly spaced queries for larger T.
Use `--check-rows 0` for all rows.

Timings cover prepared kernel calls. Padding, sparse-index expansion and GBSA
metadata creation take place before profiling. The runner alternates the order
across three msprof rounds, with ten warmups and twenty measured calls per
implementation per round. CSV latencies are the medians of the round means in
microseconds. A speedup of at least 0.90 (GBSA time divided by QSA time) meets
the target. A correctness failure skips timing for that case. Inspect the
individual status and detail fields on a nonzero exit. Each invocation writes
a fresh CSV, so use a distinct output name for a rerun you want to keep.
