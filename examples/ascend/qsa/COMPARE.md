# QSA Generalization Validation and GBSA Comparison on Ascend 950

Use the TileLang build for this branch together with an ops-transformer GBSA,
GBSA Metadata, and `cann_ops_transformer` extension supporting the Ascend 950
BF16 TND/TND, `D=256`, `[1, 4]` block, and residual-tail paths (baseline API at
ops-transformer commit `6945d02b7`). The stock CANN 9.3.0-beta.1 Metadata
operator only accepts `PA_BBND` and cannot run TND GBSA comparisons (use
`--no-gbsa` if you only want to validate QSA functional correctness).

From the repository root with CANN and the custom OPP environment active:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD"

# 1. Run the full 13-case self-test matrix (C01-C13, single dynamic compile)
python3 -m examples.ascend.qsa.compare_gbsa --csv qsa-gbsa.csv

# 2. Run selected preset cases
python3 -m examples.ascend.qsa.compare_gbsa --cases C01 C03 C09 --csv smoke.csv

# 3. Run one uniform packed batch (--shape B Q KV)
python3 -m examples.ascend.qsa.compare_gbsa --shape 4 1 1048575 --csv kv4m.csv

# 4. Run one or more custom heterogeneous batches in a single dynamic-compile run
python3 -m examples.ascend.qsa.compare_gbsa \
  --requests 1:24577,14:32770,15:65539 16x1:65536 2x4096:131072 \
  --csv custom.csv

# 5. Fast accuracy-only sweep (skip msprof benchmarking; add --no-gbsa if GBSA is unavailable)
python3 -m examples.ascend.qsa.compare_gbsa --check-only --check-rows 0 --cases C01 C02 C03
```

## Notes

- **Preset matrix (`C01`–`C13`)**: Covers `24K <= s_b <= 1M`, `1 <= T <= 32K`,
  and `1 <= B <= 128`, including lower bounds, all tail lengths (`0/1/2/3`),
  heterogeneous non-4-aligned batches, large-batch decode (`B=32/128`), and
  chunked prefill up to `T=32K` / `s_b=1M`.
- **Single-compile dynamic reuse**: Every invocation compiles one dynamic QSA
  kernel (`compiled_once=True`) for `--aic-count` (default `28`; also supports
  `32` and `36`) and reuses it across all cases in the run.
- **Accuracy**: CPU FP32 reference reconstructs selected tokens independently
  from the causal block lists and checks `|out - ref| <= 0.01 + 0.01 * |ref|`
  for both QSA and GBSA. Default checks every row for `T <= 128` and 64 evenly
  spaced rows (including endpoints) for `T > 128`; use `--check-rows 0` for
  all-row verification.
- **Performance**: Prepared kernel-only `msprof` timing (padding, index split,
  and GBSA metadata creation are outside the timed region). Runs `--rounds`
  (default `3`) alternating rounds of `--warmup` (`10`) warmups and `--repeat`
  (`20`) measured calls, reporting the median of round means in microseconds.
  A case passes when `GBSA / QSA >= --target-speedup` (default `1.0`).
