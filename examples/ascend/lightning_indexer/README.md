# Lightning Indexer Sparse Backward (Ascend 950DT / `dav-3510`)

TileLang Ascend implementation of Operator 1 (`lightning_indexer_sparse_bwd`) for
DeepSeek-V4.1-Flash CSA2 / DSA sparse training backward propagation (v1.0 spec):

- **Static compile-time dimensions (v1)**: `H = 32`, `D = 128`, `K = 512`, `BK in {64, 128}`
- **Multi-SKU Core Support**: `num_cores in {28, 32, 36}` (**default `32` AIC / 64 AIV** per acceptance standard; supports 28 AIC / 56 AIV test devices and 36 AIC / 72 AIV full-chip 950DT)
- **Dynamic runtime dimensions (no JIT recompile)**: `B`, `S_q`, `S_k` (supports `S_q != S_k` and Packed THD via `B = 1`), runtime `scale`
- **Dtypes**: `q / k / dq / dk: bfloat16`; `w / dw / dI: float32`; `indices: int32` (valid in `[0, S_k)` or `-1` sentinel)

## Roofline Summary (`B=1, S_q=32768, S_k=16384, H=32, D=128, K=512`, 32 AIC / 64 AIV)

- **Useful Cube FLOPs**: `412.32 GFLOPs` -> Cube compute floor = `0.954 ms` (@ 432 TFLOPS BF16)
- **Useful Vector FP32 FLOPs**: `2.15 GFLOPs` -> Vector compute floor = `0.080 ms` (@ 27 TFLOPS FP32)
- **Cold HBM (0% L2 reuse, 4 TB/s nominal)**: `3.39 ms` (1x atomic) to `5.54 ms` (2x RMW) — HBM-bound (`MFU <= 17–28%`)
- **L2-Resident `K` (4.2 MB) + `dK_fp32` (8.4 MB) = 12.6 MB << 128 MB L2**:
  - HBM streaming floor drops to `0.175 ms` (`0.70 GB`)
  - AIV MTE3 `dK_fp32` atomic-to-L2 port model floor = `1.938 ms` (64 AIV @ `ub_to_l2_rate=42`, 1.65 GHz) -> **Atomic/L2 Bandwidth-Bound** (Cube MFU ceiling ~`49.3%`)
- **Key Optimizations Targeting This Roofline (V4 5-Stage Prefetched SIMD-Guided Pipeline)**:
  1. **Exclusive L2 Pinning**: `l2_cache_ctrl="NORMAL_FV"` on `K` (4.2 MB) and `dK_fp32` (8.4 MB) paired with streaming no-allocate `NOTALLOC_DROP` on zero-reuse loads (`Q`, `W`, `Indices`, `dI`) and `NOTALLOC_CI` on streaming stores (`dQ`, `dW`) to eliminate L2 cache pollution.
  2. **Flattened 1D Cross-Query 5-Stage `T.Pipelined(q_count * NUM_K_TILES, num_stages=5)`**: Stage 0 prefetches `Q` (2 stages ahead of `GEMM1`, eliminating the cold-HBM query-boundary stall), loads `Indices`/`dI`, and runs `analyze_idx_contiguity`; Stage 1 executes pure `MTE2` Gather using `meta_ub` already ready from Stage 0 (zero `PIPE_V -> PIPE_S` stall and zero `MTE3` HOL block); Stage 2 ships `K_ub -> K_l1` + `GEMM1` + `W` prefetch; Stage 3 runs `256B SIMD` + `GEMM3/2` + direct AIC FixPipe FP32->BF16 `dQ` store; Stage 4 executes `MTE3` Atomic Scatter + `dW` store. All buffer versions are powers of 2 (`& 1 / & 3 / & 7`, zero modulo, zero `V_MTE2` flag spill).
  3. **256B SIMD Contiguity & Validity Analyzer (`analyze_idx_contiguity`)**: emits single `16 KiB` MTE2 / `32 KiB` MTE3 bursts on 64-row contiguous half-tiles (`BK=128`), 2-burst `4 + 60`-row (`1+15 KiB` / `2+30 KiB`) fast path on `sink_tokens=4` + contiguous window tiles, 4-way register-prefetched branchless `scatter` loop on pure random indices, O(1) skip on all-invalid tiles, and 16/8/4-row fallback on general mixed tiles.
  4. Direct AIC FixPipe `dQ_l0c -> dQ` (`NOTALLOC_CI`) removes AIV `dQ` cast/store overhead; `T.set_atomic("add", fp32)` is scoped to Stage 4 `dK_fp32` scatter, plus pure-SIMD TileLang auxiliary kernels (`zero_fp32`, `cast_fp32_to_bf16`) for zero `aclnn` OPP dependency.

## Usage

```bash
# 1. CPU reference & FP64 gradcheck (no NPU required)
python examples/ascend/lightning_indexer/verify_lightning_indexer_bwd.py --mode cpu-ref

# 2. Offline CANN compilation to .o + CCE audit (28 / 32 / 36 AIC)
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python examples/ascend/lightning_indexer/verify_lightning_indexer_bwd.py --mode compile --compile-cores 28 32 36

# 3. Full Ascend 950DT NPU accuracy suite (wrap in npu-guard on shared 950DT)
python examples/ascend/lightning_indexer/verify_lightning_indexer_bwd.py --mode npu --num-cores 32 --bk 64
python examples/ascend/lightning_indexer/verify_lightning_indexer_bwd.py --mode npu --num-cores 28 --bk 128

# 4. NPU accuracy + AB/BA paired msprof_detail benchmark
python examples/ascend/lightning_indexer/verify_lightning_indexer_bwd.py --mode npu --num-cores 32 --bench

# 5. Single-operator run-only mode (no accuracy verification) for msprof op pipeline capture
#    Use --only-main-kernel to capture ONLY the core Cube+Vector kernel (skipping zero/cast aux kernels)
#    Use --seq-q 4096 if a smaller trace is preferred, or --sink-tokens 0 to isolate sink atomic contention
msprof op --output=./msprof_li_bwd -- python examples/ascend/lightning_indexer/verify_lightning_indexer_bwd.py \
  --mode profile --num-cores 28 --bk 128 --only-main-kernel --warmup 0 --repeat 1
```
