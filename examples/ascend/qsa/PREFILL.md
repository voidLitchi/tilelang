# 4096-query prefill 的性能目标

以下均为 prepared 单算子，batch=1、queries=4096、kv=32768、seed=7、静态形状。带宽采用十进制 TB/s，时间为单次 kernel 的耗时。

## 950PR 上的分支差异

28 AIC 950PR 上，用户实测 old 为 7558.09 µs，exp 为 8006.29 µs，增加 448.20 µs（5.93%）。GBSA 为 6880–7055 µs。按同一轮 GBSA 时间除以 0.90，目标为 7644–7839 µs。要覆盖观测到的最快 GBSA，应以 7644 µs 为第一阶段目标。

对照 `qsa-old@919dd815` 与 `qsa-exp@8915a637` 的 native 编译器，在相同 QSA 源码、shape 和 fast-math 下生成 AscendC。物理任务数为 4096×2=8192。每个任务有 17 个 128-token tile；`NI != 2` 在本例恒为真。exp 的 prefill 源码与双 tile 规避前的产物 SHA-256 一致。

下表按生成源码展开一个任务的循环、条件与 AIV 分支计数。这里所有 quad 基址均可走 4-row DMA，包括映射至 0 的填充槽。循环上界含流水首尾，调用数是执行次数，均不等于 stall 周期。

| 每个任务 | old | exp |
|---|---:|---:|
| AIC / AIV0 / AIV1 外循环上界 | 26 / 22 / 22 | 25 / 22 / 24 |
| AIC 的 MMAD | 68 | 68 |
| 每个 AIV 的 GM→UB / UB→L1 | 578 / 85 | 578 / 85 |
| AIC 跨核 arrive + wait | 244 | 312 |
| 每个 AIV 跨核 arrive + wait | 122 | 156 |
| 每个 AIV 核内 MTE2→V 通知 / 等待 | 68 / 68 | 85 / 85 |

exp 将两个 AIV 分开排程。PV 的临时 UB 使用单版本，向量累加读取之后各 AIV 增加 V→FIX 归还通知，后续 FIX 写回需等待；Valid 搬入后也出现单独的 MTE2→V 同步。相同 KV 搬运量现在经过不同的依赖链。这些边可能承担正确性职责，耗时归因应看等待前后的空闲周期。

另有一个独立变量。提交 `577528a4` 更换了 msprof 的逐轮 L2 清理实现，old 把 256 MiB 缓冲用 WTS_FV 写一遍，exp 用 NORMAL_FV 读进 UB。清理 kernel 的时间会被过滤，留下的 cache 状态仍可能影响下一次 QSA。这两项变化的时间占比需要实机交叉实验。[Profiler 变更](https://github.com/tile-ai/tilelang/pull/3534)

### 分离编译产物和测量工具

在服务器已有的两个独立构建中分别运行下述命令；各自的 native 库与 Python 源码须匹配。使用本分支的同一份 `examples/ascend/qsa`，旧构建的仓库本身没有该用例。先在包含这两个 Git 对象的仓库导出驱动：

```bash
git show 919dd815:tilelang/profiler/msprof.py >/tmp/qsa-msprof-write.py
git show 8915a637:tilelang/profiler/msprof.py >/tmp/qsa-msprof-read.py
```

在每个已激活的构建根目录执行。脚本只在当前 Python 进程替换逐轮采样驱动，固定 50 轮。第二遍倒序运行，保留每一行的 GBSA 时间作为同机对照。

```bash
for flush in write read read write; do
  PYTHONPATH="$PWD" python3 - "/tmp/qsa-msprof-$flush.py" <<'PY'
import importlib.util, sys, tilelang
from examples.ascend.qsa.compare_gbsa import main
path = sys.argv[1]
name = "tilelang.profiler.msprof"
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
print(tilelang.__file__, path, flush=True)
sys.argv = ["compare", "--batch", "1", "--queries", "4096", "--kv", "32768",
            "--scope", "prepared", "--repeat", "50", "--no-check"]
main()
PY
done
```

固定清理方式后比较 old/exp，得到编译产物一侧的差值；固定构建后比较 read/write，得到采样条件一侧的差值。若两列结果有交互，分别报告。同一清理方式下若 exp 仍慢，优先比较两个 AIV 的完成时间、FIX 等待 O_tmp 复用的间隙、Valid 到 softmax 的等待，再看实际 GM 读字节与 L2 命中。单次采集入口见 [对标说明](COMPARE.md)。

## 作者版的对照变量

`kernel_gm2l1.py` 的默认值仍为 M=32。Cube 路径为每 tile 的 32 个四元块各取一个基址，供 K 与 V 共用。生成代码中每次搬 4×256 BF16，源行距 1024 B，NZ 目标按整块 128 行设 pitch，相邻四元块的目的起点相差 64 个 BF16 元素。K、V 各 32 条 GM→L1 ND2NZ 取代 KV 的 GM→UB、UB 内 ND→NZ 和 UB→L1。每任务仍发出 1088 条 KV DMA；短 DMA 的发射速度、依赖标量地址的延迟和 AIC MTE2 占用都值得单独采集。AIC 源码每 tile 出现 32 个基址读取，两个 AIV 分支中也各有一组索引表达式；AIV 中的死读是否留在机器码应以计数器为准。

两个作者变体均先清零 O，之后每个 PV 写入双版本 O_tmp 并执行 `O = O * alpha + O_tmp`，第一轮也走同一条边。原 baseline 在第一轮直接写 O，O_tmp 只有一个版本。因而 baseline 到 author-cube 的差值同时包括累加、流水和 gather 的变化。author-aiv 到 author-cube（相同 M）的对照保留了共同的累加方式。M=16 则把每组 12 头的物理行数从 32 降到 16，矩阵工作量和按行进行的向量工作随之减半，KV 请求量相同。原文件的 `dynamic_shape` 参数未构造符号维度；本次接入为 Q 和补齐后的 K/V 行数建立符号，静态默认保持作者的规格。

## 分层 roofline

每个 query 有 512 个完整四元块，右端可见尾长 0、1、2、3 各出现 1024 次。平均有效 token 数为 2049.5，QSA 固定处理 2176 槽。24 个有效 Q 头分给两个 KV 头，单个 KV 头的 12 行在 baseline 中扩到 M=32。

| 量 | 每次 prefill |
|---|---:|
| 有效 QK+PV，按 FMA=2 | 4×4096×2049.5×24×256 = 206.309 GFLOP |
| baseline 实际矩阵工作，含两类填充 | 4×4096×2176×2×32×256 = 584.116 GFLOP |
| 有效 KV 请求字节，两份 BF16、两个 KV 头 | 17.192 GB |
| baseline 全部槽位的 KV 请求字节 | 18.254 GB |
| 物理 K、V 两个完整张量总容量 | 67.109 MB（64 MiB） |
| padded Q 读 + O 写；Indices + Valid 各读一次 | 268.435 MB；71.303 MB |

同一 K/V 可跨 query 重复命中 L2。若每张索引、掩码只从外存读一次，令 h 为按字节计算的 KV 请求片上满足比例，baseline 的两项下界为

```text
T_cube = 584.116 GFLOP / BF16 Cube 峰值
T_mem  = [0.339739 GB + 18.253611 GB × (1 − h)] / 外存峰值带宽
T ≥ max(T_cube, T_mem, T_L2→核, T_片内搬运, T_Vector)
```

前两项允许计算和搬运充分重叠。该估算忽略 cache-line 放大和索引重读。冷启动时，即便理想复用仍需取入用到的物理 KV。PR 28 核档有 112 MB L2，DT 有 128 MB；64 MiB 的 KV 容量允许大量复用，流式 Q/O、元数据和稀疏访问顺序会改变实际命中率。白皮书表 3-1 给出的 BF16 Cube 峰值与外存带宽分别为 PR 28 核 378 TFLOPS、1.4 TB/s，DT 36 核 486 TFLOPS、4 TB/s。[华为《昇腾 950 NPU 架构白皮书》](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)

| 理论耗时或下界（ms） | 950PR 28 AIC | 950DT 36 AIC |
|---|---:|---:|
| 只按有效模型 FLOP 的绝对计算下界 | 0.546 | 0.425 |
| M=32、2176 槽位的 Cube 下界 | 1.545 | 1.202 |
| M=16、2176 槽位的 Cube 下界 | 0.773 | 0.601 |
| M=32，h=0，计算和外存两项取最大 | 13.281 | 4.648 |
| M=32，h=50% | 6.762 | 2.367 |
| M=32，h=80% | 2.850 | 1.202 |
| M=32，KV 理想复用（约 0.407 GB 外存总量） | 1.545 | 1.202 |

DT 的 32 / 28 核档 BF16 Cube 为 432 / 378 TFLOPS，M=32 的计算下界分别为 1.352 / 1.545 ms，外存项按同一 4 TB/s 计算。理想复用行给出当前算法在只考虑两项屋顶时的乐观极限；它要求片内也能供数。例如 PR 若要逼近 1.545 ms，仅 KV 的 L2/存储层到核就需约 11.8 TB/s，且 baseline 还要经 UB 重排及 UB→L1 再搬一遍。架构白皮书的外存带宽不足以推导这几条通路的持续吞吐。

用户提供的 PR 数据对应：

| 实现 | 有效 BF16 Cube MFU | 有效 KV 吞吐 | 含填充的 QSA KV 吞吐 |
|---|---:|---:|---:|
| qsa-old | 7.22% | 2.275 TB/s | 2.415 TB/s |
| qsa-exp | 6.82% | 2.147 TB/s | 2.280 TB/s |
| GBSA | 7.74–7.93% | 2.437–2.499 TB/s | 需按 GBSA 自身 tiling 计算 |

exp 的实际矩阵工作量折算为 19.3% Cube 峰值。2.280 TB/s 已超过 PR 外存峰值，说明请求字节有复用。在上面的 Q/O 和索引假设下，exp 至少约 40% 的 KV 请求字节在片上满足；这只是带宽预算推得的下界。实际外存利用率请用 PMU 的外存读写总字节除以（耗时 × 1.4 TB/s），并核对计数器汇总范围。再结合 L2 读命中、AIC/AIV MTE2、AIV MTE3、Vector 和 FIX 的占用、标量地址生成及等待，才能判定瓶颈在外存、片内中转、DMA 发射或同步。若 direct gather 后外存读量相近、UB 中转消失且耗时下降，就能量化这一优化的实际价值。
