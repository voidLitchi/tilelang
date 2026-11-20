# 在 Ascend 950 对标 QSA 与 GBSA

## 部署和 smoke test

**版本限制**：在 ops-transformer 的 `v9.3.0-beta.1` 标签上，950（A5）的 GBSA 只支持 PA_BBND KV、D=128、[1,128] 稀疏块、最多 256 个块且不支持隐式尾块。不能靠换 layout 或重排 K/V 做本例的等价对标。A2/A3 的规格不适用于 950。这里依赖的是安装的算子实现，不单由 CANN 的版本字符串决定；更新 Python extension 而仍加载旧主算子/Metadata 也不够。使用匹配工具链构建支持下述配置的两个算子与 extension，或换到已部署该版本的环境。先用 `python -m examples.ascend.qsa.compare_gbsa --probe-only` 检查实际加载的 Metadata；通过后仍须运行精度检查。

本分支基于 QSA PR #24 的 `7f546ecd`，叠加 tile-ai/tilelang #3323 的两个提交（合成后为 `1fb1cec3`），并在 QSA 的双 tile 路径关闭 stage offset。使用配套的驱动、CANN、PyTorch/torch_npu、TileLang 的 Ascend 后端，并安装 ops-transformer **及其 cann_ops_transformer torch extension**，包括 GenericBlockSparseAttention 和 GenericBlockSparseAttentionMetadata 两个算子。脚本按 ops-transformer `6945d02b7` 的 Python API 编写。CANN 环境须先 source，确保自定义 OPP 和 extension 指向同一次安装。切换构建后应核对实际加载的 libtilelang.so。

将本分支推送后，在 950 服务器的干净 TileLang checkout 中执行以下命令。将 `origin` 替换为实际接收该分支的远端。脚本已包含在分支中，不需要 zip、bundle 或再次 cherry-pick。

```bash
git fetch origin refs/heads/codex/qsa950-compare
git switch --detach FETCH_HEAD
git submodule update --init --recursive
# 在服务器专用环境中按仓库 Ascend 指南重建；离线环境按本地依赖调整
USE_ASCEND=ON USE_CUDA=OFF python -m pip install -v .
```

从源码根目录运行；首次调用可能编译 torch extension 与 TileLang kernel，不计入结果。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD"
python -c 'import torch_npu, torch, cann_ops_transformer as c; print(torch.npu.get_device_name(0), c.ops.generic_block_sparse_attention_metadata, c.ops.generic_block_sparse_attention)'
python -m examples.ascend.qsa.compare_gbsa --check --check-rows 0
python -m examples.ascend.qsa.compare_gbsa --batch 1 --queries 1 --kv 65539
python -m examples.ascend.qsa.compare_gbsa --batch 8 --queries 1 --kv 65536
python -m examples.ascend.qsa.compare_gbsa --batch 1 --queries 4096 --kv 32768 --check-rows 16
python -m examples.ascend.qsa.compare_gbsa --batch 1 --queries 1 --kv 262144 --dynamic --no-check --scope prepared --repeat 50
```

`--kv` 是每个请求包含当前 query 的 KV 长度，`--queries` 是每请求的右对齐 query 数，不是 batch。`--batch 128 --queries 1 --kv 65536` 与 PR 共享同一 KV 池的“128 decode”不同，独立请求仅 BF16 K+V 就需 16 GiB，先用小 batch。要留出 profiler、输出和算子 workspace。打印当前设备和 seed；记录 CANN/OPP/TileLang revision、卡型号和重复次数，尽量独占卡。

## PV 同步精度回归

在同一台 950 上，从 TileLang 根目录分别执行：

```bash
python3 -m examples.ascend.qsa.diagnose_pv_sync --all
python3 -m examples.ascend.qsa.compare_gbsa --check --check-rows 0
python3 -m examples.ascend.qsa.compare_gbsa --check --check-rows 0 --dynamic
```

第一条依次执行 `test_qsa.py` 的 5 项回归和 `run.py` 的 3 个默认规格（各跑静态、动态两版），覆盖双 tile 和长流水。任一断言失败会中止。后两条的 QSA 编译规格固定为 2176 个槽位（17 tiles），分别将 TileLang、GBSA 与同一输入的 FP32 标杆比较，默认检查 2 个请求各 4 个 query 的全部 8 行。要单独跑双 tile 手工可验输入，执行 `python3 -m examples.ascend.qsa.diagnose_pv_sync`。

当前规避只在 `NI == 2` 时关闭 stage offset，保留 AutoSchedule、自动分配 flag 和自动插同步。上述三条命令均直接执行原生生成代码。可选的 `--pv-ready` 是旧版诊断工具，在已核对的新双 tile 代码上识别原生 FIX 8/29，在 17-tile 代码上识别原生 FIX 12/28。要复查旧版 A/B，请使用 `a6a20322` 的双 tile 单例，分别省略和加上 `--pv-ready`；在那里开关对已核对的旧源码补 FIX 15/31 及两处 AIV 等待。此前上板通过的是这个加同步的旧单例。

## 在 tile-ai/tilelang/main 单独复现

将本目录的 `repro_stage_guard.py` 复制到已构建的上游 TileLang 源码根目录，在该根目录执行。复核提交为 `4ed9b9edb5c9a1e05f52b3bd45e7116fea7ec9ab`。脚本只依赖上游 Ascend DSL，无需 QSA、GBSA 或 #3323。

```bash
PYTHONPATH="$PWD" python3 repro_stage_guard.py --dump-dir /tmp/stage-guard --check-ir
# 独立运行设备检查。先跑同 stage 对照，再跑错位变体，逐元素精确比较。
PYTHONPATH="$PWD" python3 repro_stage_guard.py --run --repeat 100
# 增大 DMA 长度，改变两条 pipe 的相对时序
PYTHONPATH="$PWD" python3 repro_stage_guard.py --run --width 1024 --repeat 100
```

第一条在该提交上预期以断言失败退出，并留下两份 PrepareMultiBuffer IR 及两份最终 AscendC。相同 stage 的两个条件可安全共享。错位变体中，`x` 的条件在 stage 1 取样，`y` 在 stage 0 取样，两者仍被映射到同一条件。最终 `stage1.asc` 的物理迭代 1 搬入第二行后，搬出同一 UB 前缺少 MTE2→MTE3 同步；物理迭代 2 的握手只覆盖第一行。

默认输入每行 8 个 FP32。第 0 次运行使用全 0 和全 2 两行，期望 `C == A`。第二行暂存 UB 预填为 0，若搬出先于第二行搬入完成，实得 0、期望 2。设备报错包含 stage、轮次、行列及数值。单次设备结果取决于 DMA 时序，IR 检查可独立复现。

## 公平性与计时

### 作者版本和单次采集

从本分支根目录运行。每条命令直接使用同一 seed、同一组 Q/K/V、选块、尾块和 FP32 标杆。默认 `baseline` 保留原来的调用。作者版本来自 `8ccb1d34` 的 `kernel_gm2l1.py`。

```bash
shape=(--batch 1 --queries 4096 --kv 32768)
for variant in baseline author-aiv author-cube; do
    python3 -m examples.ascend.qsa.compare_gbsa "${shape[@]}" --qsa-variant "$variant" --scope prepared --check --check-rows 16
done
python3 -m examples.ascend.qsa.compare_gbsa "${shape[@]}" --qsa-variant author-cube --pad-m 16 --scope prepared --check --check-rows 16
```

作者命令 `kernel_gm2l1.py --pad-m 16 --aiv-gather --stages 3` 选择 AIV 中转。对齐其 prefill 内核配置请用下面的命令，在 old、exp 的配套构建中分别使用同一份测试脚本：

```bash
python3 -m examples.ascend.qsa.compare_gbsa --batch 1 --queries 4096 --kv 32768 \
    --qsa-variant author-aiv --pad-m 16 --stages 3 --scope prepared --check --check-rows 16
```

复核时，作者独立脚本取 `[prefill-qwen3.8]` 行。它的计时输入以 seed=1 从整个 KV 池取每行 512 个完整块（2048 个有效 token），预热 30 次。对标脚本默认 seed=7，从每行因果可见前缀选块并加尾，平均 2049.5 个有效 token，prepared 预热 10 次，两者默认均采 20 次。以上命令对齐内核配置，保留 GBSA 对标的输入契约。要拆分配置收益，固定分支和 seed，依次测 author-aiv 的 M32/stages2、M16/stages2、M16/stages3，最后在 M16/stages3 下只切换 gather。

`author-aiv` 和 `author-cube` 共用作者的 PV 累加写法。先在 M=32 比较两条 gather 路径，再在同一路径比较 M=16 和 32，可以区分缩小矩阵和缩短 KV 搬运路径的收益。两种作者版本都在物理 KV 末尾补 3 行零，GBSA 和标杆继续使用逻辑 KV。prepared 提前补齐，adapter 将补齐开销计入每次调用。小 shape 可用 `--check-rows 0` 全检，也可加 `--dynamic`。

单次模式只调用一次选定的 QSA 并同步返回。它跳过 GBSA 导入、Metadata、标杆、预热、清缓存和 do_bench。输入准备仍有 torch NPU 操作，外部采集用 kernel 名筛出 QSA。单次耗时与 prepared 的清缓存时间口径不同。

```bash
python3 -m examples.ascend.qsa.compare_gbsa "${shape[@]}" --qsa-variant author-cube --qsa-once
msprof op --kernel-name=main_kernel --launch-count=1 --warm-up=0 --aic-metrics=PipeUtilization --output=./qsa-cube-pipes \
    python3 -m examples.ascend.qsa.compare_gbsa "${shape[@]}" --qsa-variant author-cube --qsa-once
```

将 variant 换成 `baseline` 可采当前原版。切换 `Memory`、`L2Cache` 指标时分别指定输出目录，核对匹配到的 kernel 和 grid（本例 8192 tasks）。多指标组合可能由 profiler 重放。分支性能归因、950PR/DT 的 roofline 及计数器判读见 [prefill 性能分析](PREFILL.md)。

同一次运行两者共享 BF16 Q/K/V。每个 query 从因果可见前缀随机无放回选至多 512 个完整四元块，升序排列，保留 0–3 个尾 token；可用 `--seed` 改变分布。这是可复现的模型契约工作负载，不是真实 indexer trace。GBSA 用 TND/TND、两头重复块号、每行实际 count、int64 cu-prefix、mask=1、residual=1、softmax_precision=0；QSA 用相同选块展开为全局 token 地址，加 batch 偏移、显式尾、补 -1 到 2176，拆 safe/valid。scale 均为 1/16。默认 smoke 含非四对齐 batch 起点和尾长 0/1/2/3。

默认 `--check` 对最多 8 个均匀分布的 query，分别和 CPU FP32 gather/softmax/PV 比较，atol=rtol=0.01；`--check-rows 0` 全检，昂贵，仅用于小形状。失败直接退出而不报告有效性能。关闭精度只适合已验证场景。普通因果行非空，不包含原 PR 的 top_k=128 全无效缺陷。

- `prepared`：索引、Q padding、GBSA metadata 都已准备，以仓内 msprof 后端计时（每轮清缓存）；只计两边 attention 的设备 kernel，不计 QSA 输出收紧。在同一固定输入上重复使用同一 metadata，仅用于隔离 attention 开销，不代表可跨请求复用。
- `adapter`：从同一张设备上的本地块号表、count 和原始 Q/K/V 开始。每次重做 QSA 展开/掩码/Q padding/输出收紧；GBSA 每次复制双头索引并重新调用 metadata、attention。预热后用同步围住整批循环的 wall time / 次数，包含 host 提交和设备工作，是热缓存连续调用的摊销时间，不是单请求 p99。不计随机数、indexer、模型投影或读盘。两种 scope 的绝对数字不要互减。
- 各自打印 `GBSA_us/TileLang_us`；相同工作量下 ≥0.90 表示该场景达到 0.9×。先通过精度，再在每个 shape、同一 scope 内比较。

## 离线编译与排障

本次在无 NPU 的 WSL 中，静态 decode/prefill、动态 2176、静态/动态 256、单 tile 128 均生成了 AscendC、dav-3510 device ELF .o 和链接后的 .aibin。实际组合是 **CANN 9.2.0-beta.2 Bisheng + asc-devkit v9.2.0 的 include/impl**，只做了编译和链接；不据此保证混用版本可上板。服务器优先使用配套版：该 PR 的 Ascend CI 指向 CANN 9.3.0 weekly 20261021。若出现 `Get different layout for KV_l1` 检查是否加载了含 NormalizeGemm 的 native 库；若报 `asc_set_l0c_copy_nz_para` 未声明或 `asc_copy_l0c2ub` 参数数不符，检查编译器**实际包含**的 C API 头，原样 beta.2 头在本机不通过。不要直接把新版头装进全局旧 CANN。

取得生成源码（无需 torch_npu 设备执行）：

```python
import tilelang
from tilelang import tvm
from examples.ascend.qsa.kernel import qsa_fwd
from pathlib import Path
with tvm.transform.PassContext(opt_level=3, config={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}):
    Path("qsa.asc").write_text(tilelang.lower(qsa_fwd(128, 65536, 2176), target="ascend").kernel_source)
```

局部复核本次混合工具链的对象编译：从 TileLang 根目录运行，KIT920 是 v9.2.0 的 asc-devkit 源码绝对路径，仅读取。该方法不替代配套运行环境。

```bash
bisheng -O2 -fPIC -std=c++20 -mllvm -cce-aicore-dcpreload-args=false --npu-arch=dav-3510 -Isrc \
  -iquote "$KIT920/include" -iquote "$KIT920" --cce-aicore-only qsa.asc -o qsa.o
```

在 17-tile lowering 中观察到 InsertSync 的 V→MTE3 flag 复用界限、FIX→V 跨迭代同步环警告。上板先做精度，尤其是动态路径、多次更换输入、不同 seed 的重复测试；若不稳定或挂起，保留生成源码与错误形状，不要以该结果报性能。本次没有仿真、没有上板延时数据。
