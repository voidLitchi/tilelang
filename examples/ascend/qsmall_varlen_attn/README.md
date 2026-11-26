# `qsmall_varlen_attn_fwd` (Ascend NPU / TileLang)

面向推荐模型场景的单查询（`seqlen_q = 1`）变长多头注意力前向算子，基于 TileLang `target="ascend"` 纯 AIV（`__vector__`）与 256B `SimdVF` 实现，目标验收平台为 **Ascend 950DT 32 AIC（64 AIV）**，同时单二进制零重编译兼容 **Ascend 950PR 28 AIC（56 AIV）**。

---

## 1. 算子规格与设计摘要

### 接口签名
```python
from examples.ascend.qsmall_varlen_attn import qsmall_varlen_attn_fwd

out, lse = qsmall_varlen_attn_fwd(q, k, v, cu_seqlens_k, sm_scale=None, num_aiv=None)
```

| 项目 | 规格 |
| --- | --- |
| 输入 `q` | `[N, 56, 32]`，`bfloat16`，contiguous row-major |
| 输入 `k`, `v` | `[total_k, 56, 32]`，`bfloat16`，contiguous row-major |
| 输入 `cu_seqlens_k` | `[N + 1]`，`int32`，第 `i` 个样本区间为 `[cu_seqlens_k[i], cu_seqlens_k[i+1])` |
| 输入 `sm_scale` | `float32` 标量（默认 `1 / sqrt(32)`） |
| 输出 `out` | `[N, 56, 32]`，`bfloat16`，contiguous row-major |
| 输出 `lse` | `[N, 56]`，`float32`，自然对数尺度 `m + ln(l)` |
| 编译期固定常数 | `H = 56`（MHA，`n_rep = 1`）、`D = 32`、无 causal mask / dropout / GQA |
| 运行期动态参数（零重编译） | `N`、`total_k`（`int64` 偏移防 `int32` 溢出）、`seqlen_k[i]`、`sm_scale`、`grid_n`（AIV 核数） |
| 边界语义 | 空序列（`seqlen == 0`）严格输出 `out = 0.0, lse = -inf`；支持未覆盖 KV 缓冲区且尾块无越界读 |

### 关键实现与优化
- **纯 AIV 访存优先设计**：`Lq = 1` MHA 算术强度仅 `~1.04 FLOP/Byte`，属典型 Memory-Bound GEMV 负载；采用纯 AIV 256B `SimdVF` 消除 Cube 分形填充与核间同步开销。
- **512B Cacheline 对齐与奇偶 4-Head 配对**：每个 task 处理 `H_PACK = 8` 个连续 Head（`8 × 32 × 2B = 512B`），利用 `vcvt(PART_EVEN/ODD)` + `vmul/vmula` + `vcgadd/vcpadd` 单向量并行完成 4 个 Head 的点积，配合 `vselr` 实现零转置 Online Softmax 与原生 `vexpdif`。
- **统一三级流水与 L2 策略**：合并 Full/Tail 区块于单一 `T.Pipelined(block_n=64, num_stages=3)`（静态 UB 占用 `197.8 / 248 KiB`），拆分 `QK+Softmax` 与 `PV` 两个 VF，配合 `NOTALLOC_KEEP` 与 Strided task 调度最大化 MTE2 通道吞吐。

---

## 2. 验收、基准测试与 Profiling 使用方式

所有能力整合于 [`verify_and_bench.py`](verify_and_bench.py)（脚本内已处理相对导入，任意工作目录均可直接执行）：

### 2.1 全量精度/边界验收（T1–T5）+ Baseline 基准测试
```bash
# Ascend 950DT 32 AIC (64 AIV) 验收
python examples/ascend/qsmall_varlen_attn/verify_and_bench.py --num-aiv 64 --bench

# Ascend 950PR 28 AIC (56 AIV) 兼容验证
python examples/ascend/qsmall_varlen_attn/verify_and_bench.py --num-aiv 56 --bench
```

常用可选参数：
- `--num-aiv <N>`：指定启动 AIV 核数（默认自动检测；在 36 AIC 测试机上请传 `--num-aiv 64` 对齐 32 AIC 验收口径）
- `--skip-t5`：跳过 `total_k = 1,250,000` 的 `int64` 防溢出测试（节省约 9 GB HBM）
- `--warmup <W> --iters <I>`：自定义计时暖机与迭代次数（规格默认 `warmup=10, iters=50`）
- `--sweep`：扫描多组 `(block_n, num_stages, l2_cache_ctrl)` 调优配置

### 2.2 实卡 `msprof` 管线占比与硬件流水图采集
通过 `--profile-only` 仅执行标准基准形状（`N=1280, L=578`），跳过 T1–T5 与静态审计开销：

```bash
# 1) 实卡 PipeUtilization + HBM 带宽 + Timeline（产出 op_summary_*.csv 与 msprof_*.json）
msprof --output=./prof_qsmall --aic-metrics=PipeUtilization --sys-hardware-mem=on \
  --application="python examples/ascend/qsmall_varlen_attn/verify_and_bench.py --profile-only --num-aiv 64"

# 2) 实卡算子级 Profiling（产出 visualize_data.bin 供 MindStudio Insight 查看 Roofline/Memory/Pipe）
msprof op --output=./opprof_qsmall --aic-metrics=PipeUtilization,Memory,L2Cache,Roofline \
  --launch-count=1 --warm-up=3 \
  python examples/ascend/qsmall_varlen_attn/verify_and_bench.py --profile-only --profile-iters 1 --num-aiv 64
```

### 2.3 无 NPU 离线编译与 CCE 静态审计
```bash
python examples/ascend/qsmall_varlen_attn/verify_and_bench.py --compile-only
```

---

## 3. 测试结果

测试环境：`Ascend950DT_9592`、CANN `9.3.0-beta.1`，验收口径严格限定 `64 AIV`（32 AIC）；基准形状 `N=1280, seqlen=578, total_k=739840, H=56, D=32, bf16`，总 HBM 访存量 `5.313 GB`。

### 3.1 精度与边界测试（T1–T5）

规格要求：`out (bf16)` `rtol=1e-2, atol=1e-2`；`lse (fp32)` `rtol=1e-3, atol=1e-3`；空序列严格 `out=0.0, lse=-inf`。

| 测试项 | 覆盖内容 | `out max_abs` | `out rel_l2` | `lse max_abs` | 结果 |
| --- | --- | --- | --- | --- | --- |
| **T1 Baseline** | `N=1280, seqlen=578, total_k=739840` | `9.77e-4` | `4.24e-5` | `9.54e-7` | **PASS** |
| **T2 Dynamic & Multi-Core** | `N=1/64/1280/2048` 混测 `56` & `64` AIV，零重编译 | `9.77e-4 ~ 1.95e-3` | `3.49e-5 ~ 3.28e-4` | `4.77e-7 ~ 9.54e-7` | **PASS** |
| **T3 Varlen & Boundaries** | `seqlen=0, 1, 2, 15, 31, 32, 33, 63, 64, 65, 578, 1025` + NaN 哨兵 | `1.22e-4` | `2.59e-6` | `9.54e-7`（空序列严格 `0.0 / -inf`） | **PASS** |
| **T4 Uncovered KV** | `prefix_pad=17, suffix_pad=29` 投毒 `NaN`/`1e4` 验无越界读 | `4.88e-4` | `2.55e-5` | `4.77e-7` | **PASS** |
| **T5 `int64` Overflow** | `total_k=1,250,000`（超过 `int32_max / 1792`） | `2.38e-7` | `2.25e-8` | `4.77e-7` | **PASS** |

### 3.2 性能基准与实卡 `msprof` 硬件指标

| 配置 / 核数 | `torch.npu.Event` 中位延迟 (`warmup=10, iters=50`) | Event 等效 HBM 带宽 | `msprof` 设备端 `Task Duration` | 设备端等效 HBM 带宽 |
| --- | --- | --- | --- | --- |
| **64 AIV (32 AIC 验收)** | **`1.481 ms`** (`min=1.475 ms`) | **`3,586.4 GB/s`** | **`1.420 – 1.431 ms`** | **`3,713 – 3,741 GB/s`** |
| **56 AIV (28 AIC 950PR 兼容)** | **`1.607 ms`** (`min=1.601 ms`) | **`3,305.7 GB/s`** | — | — |

| `msprof PipeUtilization` 指标（64 AIV） | 实测值 |
| --- | --- |
| **`aiv_mte2_ratio`** | **`97.3% – 97.4%`**（有效搬运段带宽 **`3,847 GB/s`**） |
| **`aiv_vec_ratio`** | **`91.5%`**（几乎完全重叠于 MTE2 之下） |
| `aiv_scalar_ratio` / `aiv_mte3_ratio` | `7.1%` / `1.5%` |
| `aiv_icache_miss_rate` | `0.0%` |
| 静态 UB 占用 | `197.8 KiB / 248 KiB` |

---

## 4. 结论

- **功能与边界验证全部通过**：T1–T5 实卡精度裕度充裕（`out` 误差低于规格门槛 5–10×、`lse` 低于 1000×），且在动态形状与 56/64 AIV 核数切换下均实现零重编译。
- **已达硬件访存性能极限**：64 AIV 验收口径下实卡 **`aiv_mte2_ratio` 达 97.4%**、设备端等效 HBM 带宽达 **3.71–3.74 TB/s**（占 950DT 4.0 TB/s 标称带宽的 **93.5%**），计算与访存近乎完全重叠，已在目标设备 950DT 32 AIC 完成验收。
