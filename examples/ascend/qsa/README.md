# QSA Sparse Attention on Ascend

Grouped-query sparse attention for Qwen 3.8 on Ascend 950, written in
TileLang. Each query attends to `top_k` whole 4-token KV blocks chosen by
an index tensor; 24 query heads share 2 KV heads with head_dim = 256.

| Files | Purpose |
| --- | --- |
| `kernel.py` | Persistent QSA forward with compiler-managed buffer versions |
| `compile.py` | Compilation entry point for static and dynamic-shape kernels |
| `testing.py` | Input generation, contract corners, and a grouped-query reference |
| `run.py` | Correctness checks and benchmarks |
| `test_qsa.py` | Kernel regression tests |

## Kernel contract

- `top_k` must be a positive multiple of 128.
- Indices are per query and shared across KV heads. Valid slots form whole
  4-token blocks: slots `4i..4i+3` hold four consecutive tokens; only the
  last run of a query may be short. Ascending order is allowed (it improves
  L2 reuse); 4-alignment is not.
- Invalid slots hold -1 in `Indices` and 0 in `Valid`. The host passes
  clamped indices (`split_indices`), so invalid quads gather token 0 and the
  `(valid - 1) * 1e30` pen mask drops their softmax contribution.
- Q is pre-grouped per KV head and zero-padded to 32 rows (`pad_q_heads`);
  the output keeps that layout and the host slices the first group rows
  per head.
- With `dynamic_shape=True`, `num_tokens` and `kv_length` are symbolic and
  one compiled kernel serves all shapes.

On an Ascend 950PR, the kernel reaches about **7.6 ms in BF16 prefill**
(4,096 query tokens, 32,768 KV, topk = 2,176) and **308 us in BF16 decode**
(128 query tokens, 65,536 KV), or roughly 27 and 21 useful TFLOPS. The flop
counts use the 24 real query heads and the average number of valid slots
per query; timings are kernel-only with L2 flushed before each launch.

## Run

Build TileLang with Ascend support; see the [backend guide](../../../tilelang/ascend/README.md).
From the TileLang repository root, select an idle NPU with
`ASCEND_RT_VISIBLE_DEVICES`, then:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0  # Replace 0 with an idle physical NPU.
export PYTHONPATH=.
python examples/ascend/qsa/run.py            # correctness + decode/prefill benchmarks
python examples/ascend/qsa/run.py --dynamic  # one symbolic kernel serves all shapes
python -m pytest examples/ascend/qsa/test_qsa.py -q
```

`--num-tokens N --kv-length M --top-k K` benchmarks a single custom shape.

## Dynamic-shape acceptance

The baseline kernel makes `num_tokens` and `kv_length` symbolic. `top_k`, head
counts, head dimension, and the padded Q layout remain compile-time properties.
The runtime shape is a packed query row count and a packed KV row count, so it
can represent one request or several requests laid out in contiguous tensors.

The default `run.py --dynamic` command covers three acceptance shapes. These
are test points, not a three-shape whitelist. To compile once and invoke that
same kernel with all three shapes, run:

```bash
PYTHONPATH="$PWD" python examples/ascend/qsa/test_dynamic_reuse.py \
  --shapes 16:2048,128:65536,4096:32768 --check-rows 8
```

The script prints `compiled_once=True`, then runs the same kernel object for
each shape. Add `--check-rows 0` for an all-row FP32 reference check. Use
`--no-check` when only the dynamic launch contract is being tested.

The first tail-specialized experiment is exposed as `old-schedule-tail`.
It keeps the static task schedule and reads the already loaded `Valid` tile
only on local tile 16. Fully invalid quads are skipped, and a partial final
quad falls back to per-row copies. Compare it with the current baseline using:

```bash
python -m examples.ascend.qsa.compare_gbsa \
  --qsa-variant old-schedule-tail --aic-count 28 \
  --batch 1 --queries 4096 --kv 32768 --no-check --scope prepared
```

The tail-specialized schedule also supports dynamic `num_tokens` and
`kv_length` when a fixed persistent block count is supplied. Use the reuse
test to compile it once and run several shapes:

```bash
PYTHONPATH="$PWD" python examples/ascend/qsa/test_dynamic_reuse.py \
  --variant old-schedule-tail \
  --shapes 16:2048,128:65536,4096:32768 --check-rows 8
```

The dynamic old schedule requires `--aic-count 28`, `32`, or `36` through the
Python API. The comparison script selects 28 by default for old variants.
It also selects `pad-m=16` and `stages=3` for old variants, so those two flags
are optional unless an explicit incompatible value is being diagnosed.

`old-schedule-tail-late` is an experimental synchronization variant. It issues
the first 64-row tail chunk with safe clamped indices, then waits for `Valid`
before the second chunk. This keeps the precision fence while allowing part of
the tail gather to overlap the `Valid` DMA:

```bash
python -m examples.ascend.qsa.compare_gbsa \
  --qsa-variant old-schedule-tail-late --aic-count 28 \
  --batch 1 --queries 4096 --kv 32768 --check --check-rows 0 \
  --scope prepared --repeat 20
```

For a single Qwen-size dynamic invocation, use:

```bash
python -m examples.ascend.qsa.compare_gbsa \
  --dynamic --qsa-variant old-schedule-tail --aic-count 28 \
  --batch 1 --queries 4096 --kv 32768 --check --check-rows 0 \
  --scope prepared --repeat 20
```
