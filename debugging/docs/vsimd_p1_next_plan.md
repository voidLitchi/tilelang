# P1 下一阶段计划：完整 Pass、per-Region fallback、Fragment 与统一 Carrier

状态：Phase 1 已实现并提交（2026-11-02）；P0 handoff（reader/verifier/interface）已完成；Phase 2+ 待继续。

## 1. 评审后的四个决定

| 编号 | 决定 | 影响 |
| --- | --- | --- |
| D1 | 不做 VF 级 all-or-nothing；以 Region 为单元，但按依赖闭包分组 | 独立 Region 可单独 LIFTED/OPAQUE；有数据依赖的 Region 组成 group 一起回退 |
| D2 | P1 变成完整 pass；分析结果以无副作用方式写入 IR | 推荐 `SIMD_VF` SBlock annotation，body 不变 |
| D3 | 统一中间层定义一个 canonical object；`T.vsimd` 与 P1 lift 共用 | 不是每个 op 一个 TIR Call，而是 region/program 级 carrier |
| D4 | 支持 `alloc_fragment`，把 fragment 视为硬件无关的**虚拟寄存器层** | P1 只记录 storage kind/access；load/store 由 P2 决定，P1 不插入 |
| D5 | carrier 用新的最小 `VSimdProgram`，不复用 V0 fusion IR | items + reads/writes + group + payload；V0 只作历史参考 |
| D6 | item 状态与 group 状态分离 | Region `LIFTED`、Copy/Reduce `TYPED`、group `P1_LIFTABLE`；P2 可再拒绝 |
| D7 | P1 槽位在 layout/version finalization 之后，v1 只收集 VF body 顶层 item | Parallel 内的 TileOp 仍走 unsupported，不在本阶段扩 |
| D8 | P1 metadata pass 默认开启，但先做无 Parallel 快速门控；有 Parallel 才分析 | 无 Parallel 时低成本；是否保持默认 on 需过 pre-merge 编译时间门控 |
| D9 | span 以 P1 槽位的 VF body 语句索引为准；P1→P2→P3 连续运行 | 中间不跑改 body 的 pass；否则重新定位/验证 span |

**立即下一步（本分支）**：Phase 1，只做完整 P1 pass + per-Region status/span/carrier；
不同时做 fragment 语义。Phase 2 再进入 fragment，随后重写 FA。

## 2. P1 Pass 的目标形态

### 2.1 Pass 位置

```text
... LayoutInference / NormalizeBufferVersion
AscendLiftVSimd          # P1：只附加 metadata，不改 body
AscendSimdVFLowerParallel / LowerTileOps   # legacy 继续处理原 body
...
P2 PlanVSimd / P3 EmitVSimd（后续）
```

`AscendLiftVSimd` 在 Phase 1 **默认运行**（metadata-only），但先做快速门控：

1. 扫描 module，若没有 `SIMD_VF`/`T.Parallel`，直接返回；
2. 不挂 annotation、不跑 PR-A/Z3；
3. 只有存在 Parallel 时才进入 A/B/C/D/E 分析；
4. 若并行 case 的编译时间超出预算，则把默认值改为 off 或用
   `tl.ascend_simd_lift=0` 关闭。

`tl.ascend_simd_lowering` 只控制后续 P2/P3 是否启用。P1 是真正的 pipeline pass，
annotation 对 legacy lowering 没有影响；P1→P2→P3 连续运行，中间不插入改 body 的 pass。
P1 pass 不替换 `T.Parallel`，因此 legacy 仍可执行；P2/P3 后续按 group 事务替换。

### 2.2 Carrier：SBlock annotation

推荐在 `SIMD_VF` SBlock 上挂：

```text
SBlock 'SIMD_VF' annotations={
  "tl.vsimd.program": VSimdProgram(
    vf_index=0,
    provenance="lifted",
    items=[
      Region(span=[0,4), group=0, status="lifted", region=VSimdRegion(...)),
      Copy(span=[4,5), group=0, src=..., dst=..., dtype=..., layout=...),
      Reduce(span=[5,6), group=0, src=..., dst=..., dim=..., kind=...),
      Opaque(span=[6,7), group=1, kind=..., reads=..., writes=...),
      Region(span=[7,10), group=1, status="opaque", reason="..."),
    ],
    captures=..., layouts=..., assumptions=...
  )
}
```

- `items` 按源序覆盖 VF body，span 连续；
- 每个 Region 独立 `LIFTED/OPAQUE`；
- `Copy`/`Reduce` 是 typed item，不是 barrier：P1 记录逻辑 src/dst、rank、dtype、layout、
  transpose/dim/clear 等，P2/P3 决定发射或拒绝；
- Opaque 保留保守读/写 footprint，便于 P2/P3 跳过；
- body 一条语句都不改；P2/P3 只替换 planned item 的 span；
- P2 拒绝 = 不替换，原语句由 legacy 处理。

**依赖组**：per-Region 是最小 *Parallel* 单元，但 copy/reduce 的结果可能被后续 Region 使用。
如果 P2 无法处理一个 copy/reduce，不能单独把它丢掉而继续 lift 其消费者；此时应把
**依赖闭包内的 Region 组**标为 OPAQUE/fallback。这不是整 VF 回退，但也不能假装单个 Region 独立。

**为什么用 annotation，而不是 body 内 AttrStmt**：

| 方案 | 优点 | 问题 |
| --- | --- | --- |
| SBlock annotation | 无副作用；天然 per-VF；V0 已验证可跨 Simplify/NormalizeBufferVersion/LowerParallel 存活 | annotation 对象需要 FFI/reflection（V0 已做） |
| body 内 AttrStmt | 打印直观 | 改动 body/effect；legacy pass 可能当 barrier；span/顺序容易错 |
| module attr side table | 不改 body | VF 身份/生命周期不稳，容易泄漏 |

结论：优先 annotation；AttrStmt 只作为临时 debug 打印，不作为 carrier。

### 2.3 per-Region fallback

```text
VF body:
  Parallel #0  -> Region LIFTED (group g0)
  Copy         -> Copy item (typed, group g0/g1 as def-use requires)
  Reduce       -> Reduce item (typed)
  Parallel #1  -> Region OPAQUE (separate group g1)
  other stmt   -> Opaque item (conservative footprint)
```

- P1 不再因为 #1 失败而回退整个 VF；
- Copy/Reduce 是 typed item，不是 Opaque barrier；
- P2/P3 只处理 `P1_LIFTABLE` group 的 span（Region=LIFTED，Copy/Reduce=TYPED）；
- #1 和无法处理的 Opaque item 继续走 legacy；
- 若一个 VF 没有任何 LIFTED/TYPED item，仍建议挂 program（items 全 Opaque）以便诊断；
  若 VF 内根本没有 `T.Parallel`，则不挂。P2/P3 只处理 P1_LIFTABLE group，全 Opaque 等价 no-op。

两个前置接口缺口：

1. PR-A 当前返回 **per-VF** status；per-Region fallback 需要 **per-VF/per-nest** 的 legality
   状态。Phase 1 增加 `AnalyzeVSimdMemoryRegions`（VF/nest 两级 status + messages）。
2. PR-B 当前有 `nest_index`，但没有 body span。Phase 1 给每个 Region 记录
   `stmt_begin/stmt_end`（或 For 的语句索引），供 P2/P3 事务替换。

### 2.4 Copy / Reduce：逻辑 item，不是屏障

目标 FA 里会出现：

```python
T.copy(m_old, m_new)                 # shared -> fragment
T.reduce_max(S_frag, m_new, dim=1, clear=False)
T.reduce_sum(P_frag, l_cur, dim=1, clear=True)
T.copy(m_new, m_old)                 # fragment -> shared
```

P1 需要把它们记录成 item：

```text
Copy {
  span,
  src: BufferRegion, dst: BufferRegion,
  dtype, layout, transpose, annotations
}

Reduce {
  span,
  src, dst,
  dim, kind, clear,
  dtype, shape/axes, annotations
}
```

原则：

- P1 记录逻辑 rank/dtype/shape，不检查 16/32-byte 对齐、不检查物理 lane；
- P2/P3 负责把任意位宽/rank 的 copy/reduce 变成指令或拒绝；
- `T.copy` 的 layout/transpose 作为 context 保留，不在 P1 物化；
- reduce 的 `dim/kind/clear` 是语义，不是 barrier；
- 如果 P2 暂时不支持某个 copy/reduce，按 §2.3 的依赖组回退，而不是静默丢弃；
- legacy 的 tileop 限制不再约束 P1 的记录能力；P1 只是不发射。

### 2.5 需要确认的边界

1. **per-Region 最小单元 vs copy/reduce 依赖**
   - Parallel Region 是最小 lift 单元；
   - 但 copy/reduce 可能产生 Region 消费的值；
   - 若 P2 拒绝一个 copy/reduce，必须回退依赖闭包内的 Region 组，不能只丢 producer；
   - 这是“dependency-closed Region group”，不是整 VF 回退。

2. **fragment 与 shared 一样参与校验 vs 跨 VF P2 决策**
   - P1 对 fragment 的 capture/legality 与 shared 相同；
   - 跨 VF 只是 metadata，不是 legality；P1 不拒绝；
   - P2 决定拒绝或插入 staging。

3. **P1 不插入 load/store vs fragment 虚拟寄存器层**
   - P1 只记录 buffer access；
   - P2 根据 `storage=fragment` 决定寄存器移动和必要的 UB load/store；
   - 这不改变 P1 的校验。

4. **copy/reduce 任意 dtype/rank vs P1**
   - P1 只记录逻辑 shape/rank/dtype/layout；
   - 不在 P1 检查 16/32-byte 对齐、lane 宽度；
   - P2/P3 负责生成合法指令或拒绝。

### 2.6 per-Region fallback 与 fragment 数据流

反例：

```python
with T.SimdVF():
    for i in T.Parallel(N):
        frag[i] = a[i]          # Region A: 可能被判 ILLEGAL
    for i in T.Parallel(N):
        c[i] = frag[i] + 1      # Region B: 自身合法
```

如果 Region A 被 fallback 到 legacy，而 Region B 被 lift，B 读取的 fragment 值来自 A 的
legacy 执行路径。P1 不知道 P2/P3 是否会为 A 保留寄存器状态，也无法保证 B 的读取语义。

因此：

- P1 **不能只逐个 Region 独立标记 LIFTED**；
- P1 需要记录 item 级 buffer **def-use/依赖**：
  - Region/Op 的 reads/writes；
  - 同 buffer 且无中间写的 producer -> consumer 边；
  - v1 可以用 buffer 级保守依赖，index 级后续细化；
- 依赖闭包内的 Region/Op 形成一个 **group**；
- 一个 group 内任一 producer OPAQUE/ILLEGAL，整个 group 的 consumer 不能 LIFTED；
- P2/P3 只替换整组，或整组回退 legacy；
- 独立、无依赖的 Region 仍可单独 lift。

v1 的简化：

- 如果 fragment 连接了多个 Region，允许把整个 **fragment-connected group** 回退；
- 极端情况下 group 可能覆盖整个 VF，但这是依赖导致的，不是无条件的 VF fallback；
- 等 P1 的 buffer/index def-use 成熟后，再缩小 group。

P1 不需要决定 P2 的物理处理，但必须给出**逻辑依赖事实**；否则 per-Region fallback 是不闭合的。

### 2.7 v1 dependency group 算法（buffer 级）

输入：按源序排列的 items；每个 item 有 `reads[]` / `writes[]`（Copy/Reduce 的 src/dst 也计入）。

规则：

1. 每个 item 一个节点；
2. **shared/global**：
   - 对每个 buffer 维护 `last_writer`；
   - 遇到 read，连边 `last_writer(buffer) -> item`；
   - 遇到 write，更新 `last_writer(buffer)`；
   - 只做 reaching-def，不做 index 级 may-overlap；
3. **fragment**：
   - 同一 fragment buffer 的所有 item **直接 union**（保守）；
   - 原因：fragment 是虚拟寄存器层，任何 legacy/lifted 边界都可能需要 staging；
4. **buffer 身份**：按 data var；`T.view`/local view 归一到 base data var；
   同 data var 视为同一 buffer（保守），不同 alloc/param 默认不混组；
5. **Copy/Reduce**：
   - 按它的 src/dst buffer 产生 reads/writes；
   - 作为普通 item 参与上述边；
6. 用并查集得到 group；
7. group 的 P1 状态是 `P1_LIFTABLE` 当且仅当组内每个 Region 都 `LIFTED`，且每个
   Copy/Reduce 都 `TYPED`；任一 `ILLEGAL/OPAQUE/UNSUPPORTED` -> group `OPAQUE`；
   `P1_LIFTABLE` 不是最终发射许可，P2 仍可拒绝整个 group；
8. 独立 group 可分别 lift/fallback；group 内的 span 一起替换或一起不做；
9. `cross_vf` fragment 使用同一模块级图；P1 记录 group/usage，P2 v1 拒绝跨 VF group。

v1 **不做** index-level may-overlap、不做 alias Z3 查询；buffer 级 over-approximation 是允许的。
它只会让 group 变大，不会让非法/不安全的组合被放行。

Opaque item 的 reads/writes 不再共用同一集合：`BufferLoad` -> read，`BufferStore` ->
write，store 的 value/indices 中的 load 也计入 read。只有遇到未知副作用的 call
（extern/tl.*）时才退回保守的 read+write，避免把 effect 漏掉。

### 2.8 Item schema、span 与语句策略

`VSimdProgram` 的 item 不是“只放 report 字符串”；P2/P3 必须能直接消费其中的
`Buffer/Var/Layout/PrimExpr`，不能再次回到原始 TIR 做分析。

```text
Item {
  span: [stmt_begin, stmt_end)
  kind: REGION | COPY | REDUCE | BIND | ALLOC | DECL | OPAQUE
  analysis_status: LIFTED | TYPED | PRESERVE | OPAQUE
  status/effective_status: LIFTED | TYPED | PRESERVE | OPAQUE
  group_id: int
  reads/writes: [Buffer] + name
  payload: structured Map
}
```

#### span 约定

- **canonical span 基数是 VF body 顶层 item 序号**，不是 PR-A/PR-B 的 `nest_index`；
- `items[i].span_begin == i`、`span_end == i + 1` 对每个 item 强制成立；
- `tl.simdvf_scope` 是透明 wrapper，不单独占 item；
- `tl.assume` 是透明、无副作用 wrapper，归入 `program.assumptions`，不占 item；
- `AllocBuffer`、`DeclBuffer`、标量 `Bind` **必须占 item**，不能静默跳过；
- Region item 额外记录 `parallel_nest_index`；`nest_index` 只作为 PR-A/B 的分析序号保留，
  P2/P3 一律使用 `span`。

#### 顶层语句策略（v1）

| 顶层语句 | item | 状态 | 说明 |
| --- | --- | --- | --- |
| `T.Parallel` For | `REGION` | 视 PR-A/B/C/D/E | 正常 lift |
| `T.copy` / `T.reduce_*` | `COPY` / `REDUCE` | `TYPED` | 逻辑 item；P2 决定发射 |
| 标量 `Bind` | `BIND` | `PRESERVE` | 保留定义，不丢 |
| `tl.assume` | 不占 item | — | 收集到 `program.assumptions`；无副作用 |
| `tl.buffer_version` / `tl.manual_multi_buffer` | 不占 item | — | 透明 wrapper，记录到 `program.attrs` |
| `AllocBuffer` / `DeclBuffer` | `ALLOC` / `DECL` | `PRESERVE` | 必须原样保留；是否 fragment 由 P2 判断 |
| 其他 `AttrStmt` | `OPAQUE` | `OPAQUE` | v1 不解释，保守保留 |
| `IfThenElse` | `OPAQUE` | `OPAQUE` | v1 不做 CF lift；即使内部含 Parallel 也不半 lift |
| 非 Parallel `For` | `OPAQUE` | `OPAQUE` | v1 不做循环 lift |
| 顶层 `SBlock` | `OPAQUE` | `OPAQUE` | v1 不做 nested region lift |
| `While` / `AssertStmt` | `OPAQUE` | `OPAQUE` | v1 不支持 |
| 其他 `Evaluate` | `OPAQUE` | `OPAQUE` | 保守 footprint |

`IfThenElse`/非 Parallel For/顶层 SBlock 的内部如果含 Parallel，v1 **不递归 lift**，
而是整块 `OPAQUE`。这是保守策略：避免 P1 只 lift 一部分、P2/P3 无法重建控制流。

如果 VF 中存在 Parallel、但顶层枚举不到任何 `T.Parallel`（例如只出现在 serial/if
内部），P1 会 `LOG(WARNING)`，并挂一个 **全 OPAQUE** 的 program，而不是静默不挂。
`tl.buffer_version` 等已知元数据 wrapper 不会触发这种情况。

#### 结构化 payload

`Region` 的 payload 直接放 TIR 对象：

```text
domain:      [{var: Var, min: PrimExpr, extent: PrimExpr}]
accesses:    [{buffer: Buffer, indices: [PrimExpr], value: PrimExpr,
               is_write: bool, dep_axes: [int], value_axes: [int],
               value_shape: [PrimExpr], has_layout: bool, layout: Layout}]
captures:    [{buffer: Buffer, kind: string, storage: string}]
outer_vars:  [{var: Var, kind: string}]
shape_syms:  [Var]
local_views: [{view: Buffer, base: Buffer}]
layouts:     [{buffer: Buffer, layout: Layout}]
assumptions: [PrimExpr]
bindings:    [{var: Var, value: PrimExpr}]
```

`Copy/Reduce` 的 `src/dst` 也放实际 `Buffer`，并保留 dtype/shape/scope：

```text
src/dst: {buffer: Buffer, name: string, dtype: string, shape: [string], scope: string}
src_region/dst_region: {buffer: Buffer, indices: [PrimExpr], extents: [PrimExpr],
                        access_type: string}
reads/writes: 名字数组 + read_buffers/write_buffers: [Buffer]
```

`Reduce` 另外记录 `reduce_kind`、`reduce_dim`、`clear`（实际 PrimExpr，不是 args 字符串）。
`args` 仅作为 debug 字段保留；P2/P3 应使用上述结构化字段，不需要解析 args。

`program.captures`/`program.fragments` 同样放实际 `Buffer`，P2 不需要按名字重解析。

`tl.assume` 不占 item，但：

- P1 的 `program.assumptions` 记录 `{condition, anchor, anchor_end}`，其中
  `condition` 是实际 `PrimExpr`，anchor 是 item span 中的插入位置；
- PR-A/PR-B 也会收集 VF-local assume，因此“在 VF 内写 assume 帮助合法性证明”是生效的；
- **P3 契约**：如果 P3 重建 body，必须在 `[anchor, anchor_end)` 重新发射对应
  `program.assumptions` 和 `program.attrs`；如果 P3 只替换 Region span、保留原 body，
  则原 assume/buffer_version wrapper 仍在；
- `tl.buffer_version` / `tl.manual_multi_buffer` 是透明元数据 wrapper，记录到
  `program.attrs`，字段为 `{key, node, value, anchor, anchor_end}`，P2/P3 需要按
  anchor 区间保留/重建。

#### 枚举含义

| 枚举 | 值 | 含义 |
| --- | --- | --- |
| `ItemKind` | `REGION` | 一个 `T.Parallel` nest |
|  | `COPY` | 顶层 `T.copy` 逻辑 item |
|  | `REDUCE` | 顶层 `T.reduce_*` 逻辑 item |
|  | `BIND` | 标量 Bind，保留定义 |
|  | `ALLOC` / `DECL` | 顶层 buffer 分配/声明占位 |
|  | `OPAQUE` | 无法建模的顶层语句，保留保守 footprint |
| `analysis_status` | `LIFTED` | Region 单独通过 A/B/C/D/E |
|  | `TYPED` | Copy/Reduce 已记录逻辑语义 |
|  | `PRESERVE` | Bind/Alloc/Decl 等 metadata item，必须原样保留 |
|  | `OPAQUE` | 单独分析就不能 lift |
| `status/effective_status` | `LIFTED/TYPED/PRESERVE/OPAQUE` | group 事务后的有效状态 |
| `GroupStatus` | `P1_LIFTABLE` | 组内所有 item 单独可 lift；P2 仍可拒绝 |
|  | `PRESERVE` | 纯 metadata group，P2 不得替换/删除 |
|  | `OPAQUE` | 组内至少一个 item 不可 lift；整组回退 |
| `StorageKind` | `GLOBAL` | 参数/GM |
|  | `SHARED` | shared/UB |
|  | `FRAGMENT` | `local.fragment`，虚拟寄存器层 |
|  | `UNKNOWN` | 未识别的 scope（不再默认归为 global） |
| `Provenance` | `LIFTED` | 来自 TIR lift |
|  | `HANDWRITTEN` | 来自未来 `T.vsimd`（P1'） |

注意：

- `analysis_status` 是 item 单独分析结果；
- `status/effective_status` 是 group 事务后的结果；
- **metadata item（BIND/ALLOC/DECL）是 `PRESERVE`，不是 `TYPED`**；纯 metadata group
  的 `group_status=PRESERVE`，P2 **不得**把它当作可替换事务，否则会删除
  `DeclBuffer(view)` 等定义；
- 只有 `P1_LIFTABLE` group 允许 P2 事务替换；`PRESERVE` 和 `OPAQUE` 都保留原 body；
- **group 为 `OPAQUE` 时，组内所有 item 的 `status/effective_status` 都是 `OPAQUE`**，
  即使 consumer 单独分析是 `LIFTED`；这样不会出现“producer OPAQUE、consumer 仍 LIFTED”的矛盾；
- `TYPED` 不是“已经发射”，只是 P1 已记录；
- `P1_LIFTABLE` 不是 P2 的最终计划状态，P2 仍有 `PLANNED/REJECTED` 等状态。

v1 仅处理 **VF body 顶层**的 Copy/Reduce；Parallel 内部的 TileOp 仍走既有 unsupported/Opaque，
不在本阶段扩展。跨 VF fragment 只记录 use/cross_vf，module-level group 由 P2 v1 拒绝。

## 3. 统一中间层与 `T.vsimd`

### 3.1 是否需要独立 op

**需要统一中间层，但不建议为每个 vsimd op 定义 TIR Call。**

- TIR `Call` 的返回类型是标量/向量，无法表达数组值；
- 每个 value 一个 Call 会把 lane 固定，违反硬件无关；
- P2/P3 需要的是完整 region/value 图，不是零散 call。

推荐：

```text
一个 canonical IR object：VSimdProgram / VSimdRegion
两个 producer：
  1. P1 lift（TIR -> VSimdProgram annotation）
  2. T.vsimd builder / region entry（handwritten -> 同一个 VSimdProgram）
一个 consumer：
  P2 / P3 / reference expander / verifier
```

如果需要一个 TIR 层 op，定义为 **region entry**：

```text
T.VSimdRegion()  /  tl.vsimd.region
  -> 构建一个 VSimdProgram/Region，挂到 SBlock annotation
```

不要定义 `tl.vsimd.add` / `tl.vsimd.load` 这种逐 value TIR op。

结论：

- `T.vsimd` 路线存在时，**必须有独立、稳定的 canonical IR object**；
- op 只在 region 入口层定义；
- P1 lift 与 P1' 共用 verifier / expander / P2 / P3，避免两套 schema 漂移。

## 4. Fragment：P1 只做虚拟寄存器层的记录

### 4.1 P1 的语义

`alloc_fragment` 是硬件无关的虚拟寄存器层：

- 它告诉 P2/P3：这块数据应放在寄存器/片上寄存器文件，而不是 UB shared；
- P1 不决定最终是否插入 load/store，也不决定物理寄存器分配；
- P1 的 IR 里本来就只有 buffer access，没有 lowered load/store；
- 因此 fragment 在 P1 中与 shared 一样参与 capture/access/合法性分析；
- **所有校验行为不变**；fragment 访问仍然做 injectivity / RAW/WAR/WAW；
- P2 看到 `storage=fragment` 后，负责决定：
  - 寄存器间移动；
  - 何时仍需要从 UB load 或向 UB store；
  - 跨 VF 时是否拒绝或插入 shared staging。

这与前面的“fragment 不插入 load/store”并不矛盾：

```text
P1：记录 buffer access，storage=fragment；
P2：根据 storage 决定不发射 memory load/store。
```

P1 不应该在 region 中新增 SSA/RegisterValue 来替代 buffer access；那会把 P2 的物理决策
提前到 P1，也会让 P1 无法保持硬件无关。

### 4.2 P1 需要记录的 fragment 元数据

- `storage = fragment`（来自 `scope == "local.fragment"`）；
- buffer 身份、dtype、shape/axis；
- 读写 access（和 shared 同一套 capture/legality）；
- layout（如果已有）；
- **跨 VF 使用信息**：
  - `definition_vf`：在 VF 内定义时的 vf index；定义在 VF 外为 `-1`；
  - `definition_scope`：`IN_VF | OUTSIDE_VF`，用于区分“定义在 VF 外”和缺失；
  - 在哪些 VF 被使用（`use_vfs`）；
  - 是否有 VF 外的读写（`uses_outside_vf`）；
  - 是否跨 VF（`cross_vf`）；
  - 是否从定义 VF 逃逸（`escapes_vf`）。

`cross_vf` 与 `escapes_vf` 是独立判断：

- `cross_vf`：被多个 VF 使用，或定义 VF 与任一使用 VF 不同；
- `escapes_vf`：仅当**定义在 VF 内**，且被其他 VF 或 VF 外使用；
- `definition_scope=OUTSIDE_VF` 且只在一个 VF 使用：`cross_vf=false`、`escapes_vf=false`；
- `definition_scope=OUTSIDE_VF` 且在两个 VF 使用：`cross_vf=true`、`escapes_vf=false`
  （它本来就在 VF 外，不存在“从 VF 逃逸”）。

### 4.3 跨 VF fragment

P1 **不拒绝**跨 VF fragment；它只是记录事实。

为什么 P1 应该给出这个分析：

- P1 已经做模块级 scope/capture 遍历，知道 fragment 的定义 VF 和使用 VF；
- P2 若重新扫描整个 module 会产生重复分析，而且容易与 P1 的 capture 结论不一致；
- P2 需要的是“逻辑事实”，不是“是否允许”的结论。

建议 metadata：

```text
FragmentUsage {
  buffer,
  definition_vf,
  definition_scope: IN_VF | OUTSIDE_VF,
  use_vfs: [int],
  uses_outside_vf: bool,
  cross_vf: bool,
  escapes_vf: bool
}
```

P2 可以基于它：

- v1：直接拒绝跨 VF fragment；
- 后续：插入 `alloc_shared` + load/store 来桥接；
- 仍然由 P2 做物理/布局决策，P1 不预判。

P1 不把 `cross_vf` 当成 `ILLEGAL`，也不改变 Region 的 LIFTED 状态；它是 P2 的 policy 输入。

### 4.4 目标 FA 中的 fragment

当前 `parallel_fa.py` 已经在 VF 内使用：

```text
S_frag, P_frag, m_new, l_cur, alpha_frag
```

并在 shared/fragment 之间使用 `T.copy` / `T.reduce_*`。P1 应记录：

- 这些 buffer 的 `storage=fragment`；
- `T.copy(m_old, m_new)` / `T.copy(m_new, m_old)` 作为 Copy item；
- `T.reduce_max/sum` 作为 Reduce item；
- 它们与前后 Parallel Region 的 span/依赖关系。

P1 不把 fragment 访问改写成 SSA，也不决定 copy 是否变成 load/store；这属于 P2/P3。

## 5. 工作拆分

### Phase 0：基线清理

- 保留 PR-A~E；新分支基于 PR-E；
- 确认 P1 pass 槽位（NormalizeBufferVersion 之后）；
- 旧 legacy tir/tirx 问题只记录，不在本阶段修。

### Phase 1：完整 P1 pass + per-Region carrier

- 新增 `AscendLiftVSimd` pass；
- 遍历 `SIMD_VF`，按源序建立 item：
  - `T.Parallel` nest -> Region；
  - `T.copy` -> Copy item；
  - `T.reduce_*` -> Reduce item；
  - 其他无法建模的语句 -> Opaque（带 footprint）；
- 每个 Region 分别跑 PR-A/PR-B/PR-C/D/E；
- PR-A 增加 per-nest status/FFI，Region 不再共用 VF 级 status；
- PR-B/C 增加 Region/item span；
- 记录 item 的 reads/writes，建 buffer 级 def-use 依赖组；
- 失败 Region/Op -> Opaque；依赖组内 consumer 不能单独 LIFTED；
- 挂 `tl.vsimd.program` annotation；
- body 不改；
- 测试：
  - 一个 VF 两个**无依赖** Parallel，一个 LIFTED 一个 OPAQUE；
  - 一个有 fragment 依赖的 producer/consumer：producer OPAQUE 时 consumer 也必须 OPAQUE；
  - T.copy/T.reduce 进入 items，且 span/逻辑 dtype/rank 正确；
  - annotation 存在、body 逐语句不变；
  - annotation 穿过 Simplify/NormalizeBufferVersion/LowerParallel；
  - VF 内有 Parallel 但全 Opaque 时仍挂 program；无 Parallel 时不挂；
- P1 metadata pass 默认运行；legacy lowering 不变；flag 只控制 P2/P3。

**UT 验证方式调整**：

- 现有 report/FFI 单测保留，用于 PR-A/B/C/D/E 的算法细节（反例、query/cache、maps）；
- Phase 1 新增的验收应以 **Pass 输出的 IR** 为主：
  - 调用 `AscendLiftVSimd()(mod)`；
  - 读取 `SIMD_VF` 的 `tl.vsimd.program` annotation；
  - 断言 items 顺序/status/span/captures/layouts；
  - 用 `structural_equal` 或逐语句比较断言 body 未被修改；
  - 断言一个 VF 内一个 Region LIFTED、另一个 Region OPAQUE；
  - 断言 annotation 穿过 `Simplify`/`NormalizeBufferVersion`/`AscendSimdVFLowerParallel`；
- report 只作为 debug renderer，不再作为 P1 pass 的唯一验收；
- 这样可以覆盖 carrier、span、per-Region fallback 和 pass 集成，而 report 测试无法覆盖这些。

### Phase 2：Fragment 语义

- PR-B capture `local.fragment`，标记 `storage=fragment`；
- PR-A 校验完全不变（与 shared 相同）；
- P1 region 不新增 SSA/RegisterValue，也不插入 load/store；
- P1 记录 fragment 的 layout/定义 VF/use VF/cross-VF 元数据；
- P2 根据 `storage=fragment` 决定不发射 load/store；
- 结构测试；
- 不要求 VF 内融合，不要求 P2/P3。

### Phase 3：对齐 FA 用例

- 当前 `parallel_fa.py` 已改写成目标态（fragment + copy/reduce）；
- 验证 P1 program 中：
  - fragment 的 storage 标记；
  - `T.copy` / `T.reduce_*` 作为 item 出现；
  - 前后 Parallel Region 的 span 与依赖关系；
- 保持 `P_nz` layout-aware direct store；
- 结构测试 + 参考展开；
- 仍不要求 legacy 可编译。

### Phase 4：P2/P3 合并 / first PR

- P2 消费 `VSimdProgram` 的 Region/Copy/Reduce items；
- P2 为任意 dtype/rank 的 copy/reduce 生成计划或拒绝；
- P3 事务替换 item span；
- P2 拒绝 -> 不替换，按依赖组回退 legacy；
- 第一版 PR：P1 pass + fragment metadata + copy/reduce item + FA 结构 + P2/P3 stub；
- 文档和接口冻结。

### Phase 5：分支到 guard

- 扩展 item/Region 支持 `IfThenElse`；
- guard-general 的 `defined_guard` 进入 value/store；
- 复用 PR-A 的 Z3 支撑 guard 查询；
- 这是后续阶段，不在当前 PR。

## 6. 验收标准

Phase 1：

- annotation 无副作用、body 不变；
- per-Region/Op 状态正确；
- 依赖组内 producer OPAQUE 时 consumer 不 LIFTED；
- 无整 VF 无条件回退；
- 无 Parallel 的 module 不发生 annotation/Z3；
- pre-merge 编译时间门控：无 Parallel 基本零成本；有 Parallel 的增量在可接受范围；
  否则默认改回 opt-in；
- 默认 legacy 不回归。

Phase 2：

- P1 records `storage=fragment` for `alloc_fragment`；
- P1 不新增 SSA/load/store，也不改变 PR-A/B 校验；
- P1 记录 fragment 跨 VF 使用事实，但不拒绝；
- P2 才对跨 VF fragment 做拒绝/桥接决策；
- P1 program 中 fragment/copy/reduce item 可见。

Phase 3：

- parallel FA 的 fragment/copy/reduce 都进入 P1 program；
- `P_nz` 仍 layout-aware store；
- 参考展开/结构校验通过。

Phase 4：

- P2/P3 只替换 LIFTED 依赖组/Op；
- reject 时模块与 pre-emit 逐语句一致；
- 依赖组回退不会留下“producer 未处理、consumer 已 lift”的半状态。

## 7. 风险与待定

1. **Span 稳定性**：P1 后移/前移、legacy pass 重建 SBlock 时，annotation/span 是否稳定；V0 已验证部分，需要补测试。
2. **Fragment 跨 VF**：P1 只记录事实，P2 v1 拒绝；这是 policy，不是 P1 legality。
3. **Fragment 未定义读**：P1 仍按 access 记录；是否需要在 P2/P3 报 definedness，需要跟现有语义检查对齐。
4. **Annotation 与 FFI**：V0 的 reflection 可复用，但不能让 `VSimdProgram` 变成 fusion 专用 schema。
5. **legacy 共存**：P1 只挂 metadata，legacy 仍 lowering；P3 替换后要保证不双重 lowering。
6. **旧 tir/tirx 问题**：FA fallback 仍不可编译，是既有问题，不阻塞 P1 结构验证。
7. **独立 op 的边界**：只做 region entry，不做 per-value TIR op；否则会绑定 lane 类型。
8. **P1 pass 幂等性**：重复运行时必须替换而不是追加 `tl.vsimd.program`。
9. **P1 自校验失败**：builder/verifier 失败应 `LOG(WARNING)` 并标 item `OPAQUE`，
   不应让用户编译直接失败；这是 compiler bug，不是用户错误。

## 8. 结论

- P1 的下一步不是继续扩 region builder，而是把已有 A~E 分析包装成**完整、无副作用的 pass**；
- carrier 选 SBlock annotation，group fallback；
- fragment 是下一优先级，P1 只记录虚拟寄存器层，P2 决定 load/store；
- P2/P3 接入可以放在 fragment + FA 对齐之后；
- V0 `vsimd-ir` 只作为历史技术参考；不复用其 schema、代码或 fusion 结构；

## 9. Phase 1 实现记录（2026-11-02，committed）

已完成：

- `src/ascend/transform/lift_vsimd.cc`：
  - 新增 `AscendLiftVSimd` pass，默认运行；
  - 快速门控：没有 `SIMD_VF` 或没有 `T.Parallel` 时直接返回，不挂 annotation、不跑 Z3；
  - 在 `SIMD_VF` SBlock 上挂 `tl.vsimd.program`；
  - body 不变；
  - per-VF items：Region / Copy / Reduce / Bind / Alloc / Decl / Opaque；
  - item span/status/analysis_status/effective_status/group/group_status；
  - VF 顶层语句稠密覆盖，`items[i].span_begin == i`；
  - Region payload 直接放 `Buffer/Var/Layout/PrimExpr`，不再只放字符串 report；
  - Copy/Reduce 的 src/dst 放实际 `Buffer`，并保留 name/dtype/shape/scope；
  - `tl.assume` 归入 `program.assumptions`（无副作用，不占 item）；
  - `AllocBuffer`/`DeclBuffer`/标量 `Bind` 都保留 item 占位，不再静默跳过；
  - 非 Parallel For / IfThenElse / 顶层 SBlock / While / Assert 明确标 `OPAQUE`，v1 不半 lift；
  - buffer 级 def-use + fragment 同 buffer union 的 dependency group；
  - group OPAQUE 时组内所有 item 的 effective status 都是 OPAQUE；
  - fragment use/definition/cross-VF metadata；
  - Copy/Reduce 记录 src/dst buffer 的 name/dtype/shape/scope，以及 args/op；
  - fast gate 与 pass 幂等（替换 annotation，不追加）。
- `src/ascend/transform/vsimd_program.{h,cc}`：
  - 新增 typed `ReadVSimdProgram` / `VerifyVSimdProgram`；
  - P1 pass 在写 annotation 前调用 verifier；
  - 失败时 `LOG(WARNING)` 并写 `validation_error`，P2 应整 VF 拒绝；
- `debugging/docs/p1_vsimd_interface.md` 已重写为正式 P1→P2/P3 handoff；
- PR-A：
  - 新增 `AnalyzeVSimdMemoryRegions`，返回 per-VF/per-nest status；
  - `AnalyzeVF` 收集每个 nest 的 status，不再只能用 VF overall status。
- Python：
  - `AscendLiftVSimd` wrapper；
  - pipeline 在 `NormalizeBufferVersion` 之后默认调用；
  - `TL_ASCEND_SIMD_LIFT` 配置默认 True，可 opt-out；
  - `src/ascend/op/builtin.cc` 注册该 pass config option，避免 PassContext 拒绝未知 key。
- 测试：
  - `testing/ascend/transform/test_ascend_vsimd_lift.py` 15 个 pass 级测试；
  - 总计 `72 passed`。

第二轮检视修复：

- Region payload 改放实际 `Buffer/Var/Layout/PrimExpr`，不再只放字符串；
- Copy/Reduce 增加结构化 `src_region/dst_region`、`reduce_kind/dim/clear`；
- `items[i].span_begin == i`，Alloc/Decl/Bind 都占 item；
- `tl.assume` 收集到 `program.assumptions`，PR-A/PR-B 也通过 `VFAssumeCollector`
  读取 VF-local assume；
- `tl.buffer_version`/`tl.manual_multi_buffer` 透明化并记录到 `program.attrs`；
- 非顶层 Parallel 现在挂全 OPAQUE program 并 `LOG(WARNING)`，不再静默跳过；
- 新增 `ParallelOrdinalCollector`，按 PR-B 的递归 outer-Parallel 序数匹配
  `infos_by_nest`，修复 serial 包裹导致的 nest 错位；
- group OPAQUE 时所有 item 的 effective status 都降为 OPAQUE；
- BIND/ALLOC/DECL 改为 `PRESERVE`，纯 metadata group 的 `group_status=PRESERVE`，
  P2 不得替换/删除；
- `program.assumptions`/`program.attrs` 增加 `anchor/anchor_end`，P2/P3 可按区间重建；
- group id 使用 `std::map` 升序输出，避免 unordered_map 顺序随机；
- 配置关闭测试改为直接调用 P1 pass 并断言无 annotation，不再依赖会被
  `LowerOpaqueBlock` 丢弃的 `device_mod`。

第三轮检视修复：

- 增加 PRESERVE UT：注入一个标量 Bind，断言 `BIND` item 为 `PRESERVE` 且组为
  `PRESERVE`，同时 Region 仍为 `P1_LIFTABLE`；
- fragment 增加 `definition_scope=IN_VF/OUTSIDE_VF`、`uses_outside_vf`；
  `cross_vf` 与 `escapes_vf` 独立判断，VF 外定义且只在一个 VF 使用不再误判为跨 VF；
- `StorageKind` 增加 `UNKNOWN`，未知 scope 不再默认归为 `GLOBAL`；
- Opaque item 的 reads/writes 拆分为 BufferLoad/BufferStore；只有未知副作用 call
  才退回保守的 read+write；
- group id 已按升序输出；
- `test_ascend_vsimd_parallel_fa.py` 保留为 PR-E region-lift 回归；Phase1 FA
  copy/reduce/fragment 覆盖在 `test_ascend_vsimd_lift.py`，Phase2 再加端到端测试。

覆盖：

- annotation 存在、body 不变；
- Region payload 是实际 `Buffer/Var/PrimExpr`，不是字符串；
- 一个 VF 两个独立 Region：一个 opaque、一个 lifted，且不同 group；
- producer OPAQUE + consumer 单独 LIFTED 时，同 group 下两者 effective status 都是 OPAQUE；
- target FA：copy/reduce items、fragment captures、P_frag_frag、单 group；
- 无 Parallel：不挂 program；
- `tl.assume` 不占 item，program.assumptions 记录，span 仍稠密；
- annotation 穿过 Simplify；
- 跨 VF fragment metadata；
- 默认 pipeline 仍走 legacy lowering（simple 64 case）；
- `TL_ASCEND_SIMD_LIFT=False` 时 P1 pass 直接不挂 annotation（不是恒真断言）。

正式 PR code commit 统计：

```text
13 files changed, 2010 insertions(+), 26 deletions(-)
```

P0 handoff 文档（reader/verifier/interface）在**单独的 docs-only handoff commit** 中，
不进入正式 PR。

已知 Phase 1 限制：

- carrier 外层仍是 `Map<String,Any>` schema；payload 已放实际 TIR 对象，typed FFI wrapper 可后续再加；
- `IfThenElse` / 非 Parallel For / 顶层 SBlock 内部即使含 Parallel，v1 也整块 OPAQUE，不递归 lift；
- copy/reduce 只处理 VF body 顶层，Parallel 内部仍 unsupported；
- `nest_index` 是 PR-A/B 的 Parallel nest 序号；P2/P3 使用 `span`，Region item 同时记录 `parallel_nest_index`；
- `test_ascend_vsimd_parallel_fa.py` 保留为 PR-E region-lift 回归；Phase1 的 FA
  copy/reduce/fragment 覆盖在 `test_ascend_vsimd_lift.py`；Phase2 端到端 FA 另加集成测试；
- group 目前按 VF 内建；跨 VF fragment 只记录 use/cross_vf，module-level group 留给 P2 v1 拒绝；
- `P1_LIFTABLE` 不是 P2 发射许可；
- `T.copy`/`T.reduce` 的 args/dtype/shape 已记录，但尚未做 P2 级语义校验。

Phase 1 达到此次闭环目标；下一步进入 fragment 语义收尾与 P2/P3 对接。
