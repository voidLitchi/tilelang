# P1 → P2/P3 接口与交接文档

状态：Phase 1 已实现并提交（2026-11-02）。
适用对象：P2 planner / P3 emitter 的实现者。
默认 lowering 仍是 legacy；P2/P3 是 opt-in。

相关实现：

```text
src/ascend/transform/lift_vsimd.cc       P1 metadata pass
src/ascend/transform/vsimd_program.{h,cc} typed reader / verifier
src/ascend/transform/vsimd_region.{h,cc} VSimdRegion + reference expander
src/ascend/transform/vsimd_memory_legality.{h,cc} PR-A
src/ascend/transform/vsimd_capture_scope.{h,cc} PR-B
```

---

## 1. P1 pass 与开关

```text
AscendLiftVSimd
```

- pipeline 位置：`NormalizeBufferVersion()` 之后、`AscendSimdVFLowerParallel()` 之前；
- 默认开启；
- 无 `SIMD_VF` 或无 `T.Parallel` 时直接返回，不挂 annotation、不跑 Z3；
- 有 Parallel 才进入 PR-A/B/C/D/E；
- body 完全不改，只给 `SIMD_VF` SBlock 挂 annotation；
- opt-out：

```python
config={tilelang.PassConfigKey.TL_ASCEND_SIMD_LIFT: False}
```

- Pass 自己读取该 config；直接调用 `AscendLiftVSimd()(mod)` 也受它控制。

---

## 2. annotation

```text
SBlock("SIMD_VF").annotations["tl.vsimd.program"] = Map<String, Any>
```

外层是结构化 `Map`；payload 中直接放实际 TIR 对象（`Buffer/Var/PrimExpr/Layout`），
不是字符串 report。

### 2.1 program 顶层字段

| key | 类型 | 含义 |
| --- | --- | --- |
| `version` | int | 当前固定 `1` |
| `vf_index` | int | 该 VF 在 PrimFunc 中的序号 |
| `provenance` | string | 当前为 `LIFTED` |
| `items` | `[item]` | 稠密、按 body 源序 |
| `groups` | `[group]` | 按 group id 升序 |
| `captures` | `[capture]` | 该 VF 实际使用的 buffer |
| `fragments` | `[fragment]` | 该 VF 实际使用的 fragment |
| `assumptions` | `[assumption]` | `tl.assume` metadata |
| `attrs` | `[attr]` | `tl.buffer_version` 等透明 wrapper |
| `validation_error` | string | 仅在 P1 reader/verifier 失败时出现；P2 应拒绝该 VF |

### 2.2 item 公共字段

| key | 类型 | 含义 |
| --- | --- | --- |
| `kind` | string | `REGION/COPY/REDUCE/BIND/ALLOC/DECL/OPAQUE` |
| `span_begin` | int | 必等于 item 序号 `i` |
| `span_end` | int | 必等于 `i + 1` |
| `group` | int | group id |
| `group_status` | string | `P1_LIFTABLE/PRESERVE/OPAQUE` |
| `analysis_status` | string | item 单独分析结果 |
| `status` | string | group 事务后的 effective status |
| `effective_status` | string | 与 `status` 相同 |
| `reads` | `[string]` | 只用于 debug/打印 |
| `writes` | `[string]` | 只用于 debug/打印 |
| `read_buffers` | `[Buffer]` | P2/P3 应使用的实际 buffer |
| `write_buffers` | `[Buffer]` | P2/P3 应使用的实际 buffer |
| `reason` | string | OPAQUE/fallback 原因，可能为空 |

### 2.3 REGION payload

额外字段：

```text
nest_index           PR-A/B 的 Parallel nest 分析序号
parallel_nest_index   同 nest_index，显式命名
region                结构化 Region payload
```

`region`：

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
dump:        string（仅 debug）
```

注意：

- `accesses.value` 对读访问可能为空；
- `dep_axes` 只描述 access map 对 domain 的依赖；
- broadcast 用 `dep_axes` 缺轴判断，不要用 `value_shape`；
- transpose 由 `accesses.indices` 表达；
- `has_layout/layout` 是 buffer context，不是 value metadata；
- 逻辑 access 保持源序。

### 2.4 COPY / REDUCE payload

公共字段：

```text
op                 tile op 名（tl.tileop.copy / ascend_copy / reduce）
args               debug 字符串数组；P2 不应依赖
src_region          {buffer: Buffer, indices: [PrimExpr],
                     extents: [PrimExpr], access_type: string}
dst_region          同上
src                  {buffer: Buffer, name, dtype, shape: [string], scope}
dst                  {buffer: Buffer, name, dtype, shape: [string], scope}
```

Reduce 额外：

```text
reduce_kind        string，如 "max"/"sum"
reduce_dim         PrimExpr
clear              PrimExpr/bool
```

当前已知缺口：

- `T.copy` 的 `transpose` 还没有独立结构化字段，仍在 `args` 中；
- target FA 的 VF 内 copy 没有 transpose，因此不阻塞 FA；
- P2 做通用 copy 前应补 `copy_transpose` / copy annotations，或明确只支持无 transpose 的 copy。

### 2.5 BIND / ALLOC / DECL

这些是 metadata item：

```text
BIND:  {var: Var, value: PrimExpr}
ALLOC: {buffer: Buffer, storage: string}
DECL:  {buffer: Buffer, storage: string}
```

- `analysis_status = PRESERVE`
- `status/effective_status = PRESERVE`
- 纯 metadata group 的 `group_status = PRESERVE`
- **P2 不得替换/删除这些 item**；否则会删掉 `DeclBuffer(view)` 等定义。

### 2.6 OPAQUE

- `kind=OPAQUE`
- `analysis_status/status = OPAQUE`
- `read_buffers/write_buffers` 是保守 footprint；
- 非顶层 Parallel、If、serial、SBlock、未知 AttrStmt、未知 Evaluate 都会变成 OPAQUE；
- P2/P3 不替换 OPAQUE group，交给 legacy。

### 2.7 groups

```text
group: {id: int, status: string, items: [int]}
```

- `id` 按升序；
- `status` 只有 `P1_LIFTABLE / PRESERVE / OPAQUE`；
- `P1_LIFTABLE`：组内只有可 lift 的 Region/Copy/Reduce；
- `PRESERVE`：纯 metadata group；
- `OPAQUE`：至少一个 item 不可 lift，或 metadata 与普通 item 混组；
- **只有 `P1_LIFTABLE` 允许 P2 事务替换。**

### 2.8 captures

```text
{buffer: Buffer, name: string, storage: GLOBAL|SHARED|FRAGMENT|UNKNOWN}
```

### 2.9 fragments

```text
{
  buffer: Buffer,
  name: string,
  definition_vf: int,            // 在 VF 内定义时 >=0；VF 外为 -1
  definition_scope: IN_VF | OUTSIDE_VF,
  use_vfs: [int],
  uses_outside_vf: bool,
  cross_vf: bool,
  escapes_vf: bool
}
```

语义：

- `cross_vf`：被多个 VF 使用，或定义 VF 与任一使用 VF 不同；
- `escapes_vf`：仅当定义在 VF 内，且被其他 VF 或 VF 外使用；
- VF 外定义 + 只在一个 VF 使用：`cross_vf=false`、`escapes_vf=false`；
- VF 外定义 + 两个 VF 使用：`cross_vf=true`、`escapes_vf=false`。

P2 v1 对 `cross_vf=true` 直接拒绝或插入 staging；P1 不做 policy。

### 2.10 assumptions / attrs

`tl.assume` 不占 item：

```text
assumption: {condition: PrimExpr, anchor: int, anchor_end: int}
```

`tl.buffer_version` / `tl.manual_multi_buffer` 透明 wrapper：

```text
attr: {key: string, node: Any, value: Any, anchor: int, anchor_end: int}
```

`anchor` 是展平后 item span 的插入位置，`anchor_end` 是 wrapper 覆盖结束位置。

**P3 契约**：

- 如果 P3 重建 body，必须在 `[anchor, anchor_end)` 重新发射对应
  `program.assumptions` / `program.attrs`；
- 如果 P3 只替换 Region span、保留原 body，则原 wrapper 仍在；
- 不能只消费 items 而静默丢掉 assumptions/attrs。

---

## 3. item / group 状态语义

```text
analysis_status = item 单独分析结果
status/effective_status = group 事务后的结果
```

| 对象 | 值 | 含义 |
| --- | --- | --- |
| Region | `LIFTED` | 通过 PR-A/B/C/D/E |
| Copy/Reduce | `TYPED` | 逻辑 item 已记录，P2 决定 |
| Bind/Alloc/Decl | `PRESERVE` | 必须原样保留 |
| 失败 item | `OPAQUE` | 不 lift |
| group `P1_LIFTABLE` | 可替换 | P2 仍可拒绝 |
| group `PRESERVE` | 不替换 | 纯 metadata |
| group `OPAQUE` | 不替换 | 整组回退 legacy |

- group 为 `OPAQUE` 时，组内所有 item 的 `status/effective_status` 都是 `OPAQUE`；
- 只有 `analysis_status` 保留单独分析结果，用于诊断；
- `TYPED` 不是“已经发射”；
- `P1_LIFTABLE` 不是 P2 的最终计划状态。

---

## 4. span / body 契约

- canonical span 基数是 **VF body 顶层、展平后的 item 序号**；
- `items[i].span_begin == i`，`span_end == i + 1`；
- `tl.simdvf_scope` 是透明 wrapper，不占 item；
- `tl.assume` 透明，进 `program.assumptions`，不占 item；
- `tl.buffer_version` / `tl.manual_multi_buffer` 透明，进 `program.attrs`，不占 item；
- `AllocBuffer`、`DeclBuffer`、标量 `Bind` 必须占 item；
- `items` 与 body 的展平顺序一致；
- P1→P2→P3 必须连续运行，中间不能插入改 body 的 pass；
- 如果中间必须跑 body-mutating pass，P2/P3 必须重新定位/验证 span。

PR-A/PR-B 的 `nest_index` 是 **Parallel nest 分析序号**，不是 body span；
Region item 额外记录 `parallel_nest_index` 用于对账。P2/P3 只使用 `span`。

---

## 5. C++ reader API

P2/P3 不应重新解析原始 annotation。

```cpp
#include "ascend/transform/vsimd_program.h"

VSimdProgram program;
std::string error;
if (!ReadVSimdProgram(program_map, &program, &error)) {
  // 整 VF 拒绝，不要部分消费
}
```

`ReadVSimdProgram` 会：

- 检查 `version == 1`；
- 解析 items/groups/captures/fragments/assumptions/attrs；
- 检查 dense span、group 覆盖/排序、status 组合、metadata/PRESERVE、
  capture storage、fragment semantics、anchor 范围；
- 失败时填 `error` 并返回 false。

P1 pass 在写 annotation 前也调用同一个 verifier；失败会：

- `LOG(WARNING)`；
- 在 program 中写 `validation_error`；
- P2 应把该 VF 视为不可用，不要部分 lift。

---

## 6. P2 职责

P2 必须：

1. 读取并验证 `VSimdProgram`；
2. 只处理 `P1_LIFTABLE` group；
3. 跳过 `PRESERVE/OPAQUE`；
4. 为 Region 决定：
   - vector width/lane/mask；
   - layout-aware store；
   - fragment access 的处理；
5. 为 Copy 决定：
   - 普通 copy / ascend_copy / ND2NZ / 拒绝；
   - transpose/layout；
6. 为 Reduce 决定：
   - kind/dim/clear；
   - 任意 rank/dtype；
7. 为 fragment 决定：
   - register / UB / spill；
   - 是否 load/store；
   - `cross_vf` 拒绝或 staging；
8. 如果任一 item 无法处理：
   - 整个依赖 group fallback；
   - 不能只丢 producer 而 lift consumer；
9. 输出 plan（annotation 或 pass 返回值），状态至少 `PLANNED/REJECTED`；
10. 不重新跑 PR-A/B，不修改 P1 body。

---

## 7. P3 职责

P3 必须：

1. 只替换 `PLANNED` group 的 `[span_begin, span_end)`；
2. 保留 group 外原 body；
3. 保留 `PRESERVE` 的 Bind/Alloc/Decl；
4. 保留 layout annotation、local view、DeclBuffer；
5. 按 anchors 重新发射 assumptions/attrs（如果重建 body）；
6. 不重复处理已经替换的 group；
7. 失败/拒绝时保持原 body，交给 legacy；
8. 不把 P1 的 reference expander 当成最终向量 emitter。

`ExpandVSimdRegionToLoops()` 只是标量 reference expander，不是 P3 的向量 codegen。

---

## 8. pipeline / 开关

目标：

```text
P1 AscendLiftVSimd
  -> P2 PlanVSimd
  -> P3 EmitVSimd
  -> legacy AscendSimdVFLowerParallel（只处理未替换 group）
```

- P1 默认 on；
- P2/P3 必须有自己的 opt-in 开关，默认 off；
- P3 替换后，legacy 不能再处理同一 group；
- fallback group 仍由 legacy 处理；
- P1 annotation 不能进入最终 codegen 逻辑。

---

## 9. 已知限制

- 只 lift VF body **顶层** Parallel；
- If/serial/SBlock 包裹的 Parallel -> 整块 OPAQUE + warning；
- 跨 VF group 没有 module-level 合并；P1 只记录事实，P2 v1 拒绝/桥接；
- `T.copy` 的 transpose/annotations 还没有完全结构化；
- `T.vsimd` handwritten producer 尚未实现；当前只有 P1 lift；
- legacy `AscendSimdVFLowerParallel` 仍匹配 `tir.exp` 等旧名，FA fallback 可能失败；
- P1 默认 on 的 compile-time 仍需 pre-merge benchmark。

---

## 10. 测试入口

```text
testing/ascend/transform/test_ascend_vsimd_memory_legality.py  PR-A
testing/ascend/transform/test_ascend_vsimd_capture_scope.py    PR-B
testing/ascend/transform/test_ascend_vsimd_region.py           PR-C/D/E
testing/ascend/transform/test_ascend_vsimd_parallel_fa.py      PR-E FA
testing/ascend/transform/test_ascend_vsimd_lift.py             Phase1 pass / schema
```

当前：

```text
72 passed
```

`test_ascend_vsimd_lift.py` 覆盖：

- annotation 存在、body 不变；
- typed payload；
- dense span；
- 独立 Region 部分 lift；
- FA copy/reduce/fragment；
- 无 Parallel 不挂 program；
- 非顶层 Parallel 全 OPAQUE；
- serial 包裹的 nest ordinal；
- PRESERVE metadata；
- group OPAQUE 降级 consumer；
- 跨 VF / VF 外单 VF 使用；
- assumptions anchors；
- config opt-out；
- 默认 pipeline legacy。

---

## 11. 交接检查清单

P2/P3 开始前：

- [ ] 只读 `P1_LIFTABLE` group；
- [ ] 使用 `ReadVSimdProgram`，不再解析原始 TIR；
- [ ] 不把 `value_shape/value_axes` 当 access map；
- [ ] broadcast 用 `dep_axes`；
- [ ] transpose 用 `accesses.indices`；
- [ ] layouts 只作为 buffer/context；
- [ ] fragment 使用 `definition_scope/cross_vf/escapes_vf`；
- [ ] `PRESERVE` 不替换；
- [ ] group OPAQUE 整组回退；
- [ ] P3 重新发射 assumptions/attrs 或保留原 wrapper；
- [ ] 不重复 legacy lowering；
- [ ] 不把 reference expander 当最终 codegen。

---

## 12. 一句话总结

P1 只负责“逻辑事实 + 事务边界 + 可验证的 annotation”；
P2 决定“能不能做、怎么做”；
P3 决定“生成什么 IR、如何替换”。
三者之间唯一的契约是 `tl.vsimd.program` annotation 和 `ReadVSimdProgram`。
