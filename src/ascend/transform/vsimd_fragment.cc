/*!
 * \file vsimd_fragment.cc
 * \brief P1 fragment facts and fragment-only region grouping.
 */
#include "vsimd_fragment.h"

#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace vsimd {

using namespace tirx;
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

namespace {

using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using VarIntMap =
    std::unordered_map<Var, int, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using ObjectSet = std::unordered_set<const ffi::Object *>;

bool IsFragmentBuffer(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "local.fragment";
}

Array<Integer> ToInts(const std::vector<int> &items) {
  std::vector<Integer> out;
  out.reserve(items.size());
  for (int item : items)
    out.push_back(Integer(item));
  return Array<Integer>(out.begin(), out.end());
}

/*! \brief Fragments allocated by each VF (VF order, outermost VFs only). */
class FragmentDefCollector : public StmtExprVisitor {
public:
  VarIntMap definition;
  std::unordered_map<Var, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      buffers;

  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF") {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    const int vf = vf_index_++;
    for (const Buffer &buffer : op->alloc_buffers) {
      if (!IsFragmentBuffer(buffer))
        continue;
      definition.emplace(buffer->data, vf);
      buffers.emplace(buffer->data, buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  int vf_index_{0};
};

/*! \brief Fragments accessed outside every VF. */
class FragmentOutsideUseCollector : public StmtExprVisitor {
public:
  VarSet outside_uses;

  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF") {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    const bool saved = in_vf_;
    in_vf_ = true;
    StmtExprVisitor::VisitStmt_(op);
    in_vf_ = saved;
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (!in_vf_ && IsFragmentBuffer(op->buffer))
      outside_uses.insert(op->buffer->data);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (!in_vf_ && IsFragmentBuffer(op->buffer))
      outside_uses.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  bool in_vf_{false};
};

/*!
 * \brief Fragments touched inside each VF.
 *
 * The table must not depend on lifted regions: a fragment can be touched only
 * by a top-level ``T.copy``/``T.reduce`` or inside a serial loop, which is
 * outside P1's analysis unit but still a real fragment use. A fragment defined
 * in a VF but used elsewhere is not reported, because that IR is not
 * constructible (the DSL rejects a VF-local buffer used outside its defining
 * region).
 */
class FragmentUseCollector : public StmtExprVisitor {
public:
  std::vector<std::vector<Buffer>> vf_fragments;

  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF") {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    if (in_vf_)
      return; // a nested VF is not a P1 unit; do not attribute its uses
    const int vf = static_cast<int>(vf_fragments.size());
    vf_fragments.emplace_back();
    current_ = vf;
    in_vf_ = true;
    StmtExprVisitor::VisitStmt_(op);
    in_vf_ = false;
    current_ = -1;
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Record(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Record(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  void Record(const Buffer &buffer) {
    if (!in_vf_ || !IsFragmentBuffer(buffer))
      return;
    for (const Buffer &existing : vf_fragments[current_]) {
      if (existing->data.same_as(buffer->data))
        return;
    }
    vf_fragments[current_].push_back(buffer);
  }

  bool in_vf_{false};
  int current_{-1};
};

class DSU {
public:
  explicit DSU(size_t n) : parent_(n), rank_(n, 0) {
    for (size_t i = 0; i < n; ++i)
      parent_[i] = i;
  }
  int Find(int x) {
    if (parent_[x] != x)
      parent_[x] = Find(parent_[x]);
    return parent_[x];
  }
  void Unite(int a, int b) {
    a = Find(a);
    b = Find(b);
    if (a == b)
      return;
    if (rank_[a] < rank_[b])
      std::swap(a, b);
    parent_[b] = a;
    if (rank_[a] == rank_[b])
      ++rank_[a];
  }

private:
  std::vector<int> parent_;
  std::vector<int> rank_;
};

/*! \brief Data vars of every buffer a region touches. */
void CollectRegionDataVars(const VSimdRegion &region, ObjectSet *out) {
  for (const VSimdCapture &capture : region->captures)
    out->insert(capture->buffer->data.get());
  std::vector<VSimdOp> work(region->ops.begin(), region->ops.end());
  while (!work.empty()) {
    VSimdOp op = work.back();
    work.pop_back();
    for (const VSimdAccess &access : op->accesses)
      out->insert(access->buffer->data.get());
    const ffi::Optional<VSimdBufRegion> buf_regions[2] = {op->src, op->dst};
    for (const ffi::Optional<VSimdBufRegion> &buf : buf_regions) {
      if (buf.has_value() && buf.value().defined())
        out->insert(buf.value()->buffer->data.get());
    }
    for (const Buffer &buffer : op->reads)
      out->insert(buffer->data.get());
    for (const Buffer &buffer : op->writes)
      out->insert(buffer->data.get());
    for (const VSimdOp &child : ChildOps(op))
      work.push_back(child);
  }
}

} // namespace

FragmentFacts CollectFragmentFacts(const PrimFunc &func) {
  FragmentFacts facts;
  FragmentUseCollector uses;
  uses(func->body);
  facts.vf_fragments = std::move(uses.vf_fragments);

  FragmentDefCollector defs;
  defs(func->body);
  FragmentOutsideUseCollector outside;
  outside(func->body);

  std::unordered_map<const ffi::Object *, size_t> index;
  for (size_t vf = 0; vf < facts.vf_fragments.size(); ++vf) {
    for (const Buffer &buffer : facts.vf_fragments[vf]) {
      auto found = index.find(buffer->data.get());
      if (found == index.end()) {
        FragmentUsage usage;
        usage.buffer = buffer;
        index.emplace(buffer->data.get(), facts.usages.size());
        facts.usages.push_back(std::move(usage));
        found = index.find(buffer->data.get());
      }
      FragmentUsage &usage = facts.usages[found->second];
      if (std::find(usage.use_vfs.begin(), usage.use_vfs.end(),
                    static_cast<int>(vf)) == usage.use_vfs.end()) {
        usage.use_vfs.push_back(static_cast<int>(vf));
      }
    }
  }

  for (FragmentUsage &usage : facts.usages) {
    const Var &data = usage.buffer->data;
    auto definition = defs.definition.find(data);
    usage.definition_vf =
        definition == defs.definition.end() ? -1 : definition->second;
    std::sort(usage.use_vfs.begin(), usage.use_vfs.end());
    usage.uses_outside_vf = outside.outside_uses.count(data) != 0;
    usage.cross_vf = usage.use_vfs.size() > 1;
    if (usage.definition_vf >= 0) {
      for (int vf : usage.use_vfs)
        usage.cross_vf = usage.cross_vf || vf != usage.definition_vf;
    }
    usage.escapes_vf =
        usage.definition_vf >= 0 &&
        (usage.uses_outside_vf ||
         std::any_of(usage.use_vfs.begin(), usage.use_vfs.end(), [&](int vf) {
           return vf != usage.definition_vf;
         }));
  }
  return facts;
}

void GroupFragmentRegions(const FragmentFacts &facts, int vf_index,
                          std::vector<VSimdRegion> *regions) {
  const size_t count = regions->size();
  if (count < 2)
    return;
  if (vf_index < 0 || vf_index >= static_cast<int>(facts.vf_fragments.size()))
    return;
  if (facts.vf_fragments[vf_index].empty())
    return;

  DSU dsu(count);
  std::unordered_map<const ffi::Object *, int> first_region;
  for (size_t i = 0; i < count; ++i) {
    ObjectSet touched;
    CollectRegionDataVars((*regions)[i], &touched);
    // Iterate the VF's fragment list instead of `touched`: the list order is
    // deterministic (first touch in the VF), while `touched` is a pointer-hash
    // set whose order can change with ASLR. The union order determines the DSU
    // roots and therefore the group numbering.
    for (const Buffer &fragment : facts.vf_fragments[vf_index]) {
      const ffi::Object *var = fragment->data.get();
      if (touched.count(var) == 0)
        continue;
      auto it = first_region.find(var);
      if (it == first_region.end())
        first_region.emplace(var, static_cast<int>(i));
      else
        dsu.Unite(static_cast<int>(i), it->second);
    }
  }

  // Assign ids by the component's first region so the numbering follows region
  // order and does not depend on DSU roots either.
  std::vector<std::vector<int>> components;
  {
    std::unordered_map<int, size_t> root_to_component;
    for (size_t i = 0; i < count; ++i) {
      const int root = dsu.Find(static_cast<int>(i));
      auto it = root_to_component.find(root);
      if (it == root_to_component.end()) {
        root_to_component.emplace(root, components.size());
        components.push_back({static_cast<int>(i)});
      } else {
        components[it->second].push_back(static_cast<int>(i));
      }
    }
  }
  std::sort(components.begin(), components.end(),
            [](const std::vector<int> &a, const std::vector<int> &b) {
              return a.front() < b.front();
            });

  int next_group = 0;
  for (const std::vector<int> &members : components) {
    if (members.size() < 2)
      continue;
    const int group = next_group++;
    std::string opaque_reason;
    for (int index : members) {
      (*regions)[index]->group = group;
      if (opaque_reason.empty() && (*regions)[index]->status ==
                                       String(region_status::kOpaque)) {
        opaque_reason = std::string((*regions)[index]->reason);
      }
    }
    if (opaque_reason.empty())
      continue;
    // A fragment-connected group is lifted or dropped together. The region that
    // caused the group to fail keeps its own reason code so the root cause
    // stays visible.
    for (int index : members) {
      VSimdRegion region = (*regions)[index];
      if (region->status == String(region_status::kOpaque))
        continue;
      region->status = String(region_status::kOpaque);
      region->reason_code = String(reason_code::kGroupOpaque);
      region->reason =
          String("grouped with an OPAQUE fragment-connected region: " +
                 opaque_reason);
    }
  }
}

Map<String, Any> FragmentTableForVf(const FragmentFacts &facts, int vf_index) {
  Map<String, Any> table;
  if (vf_index < 0 || vf_index >= static_cast<int>(facts.vf_fragments.size()))
    return table;
  const std::vector<Buffer> &buffers = facts.vf_fragments[vf_index];
  if (buffers.empty())
    return table;

  std::vector<Map<String, Any>> fragments;
  for (const Buffer &buffer : buffers) {
    const FragmentUsage *usage = nullptr;
    for (const FragmentUsage &candidate : facts.usages) {
      if (candidate.buffer->data.same_as(buffer->data)) {
        usage = &candidate;
        break;
      }
    }
    if (usage == nullptr)
      continue;
    Map<String, Any> fragment;
    fragment.Set("buffer", Any(buffer));
    fragment.Set("name", Any(String(buffer->name)));
    fragment.Set("definition_vf", Any(Integer(usage->definition_vf)));
    fragment.Set("definition_scope",
                 Any(String(usage->definition_vf >= 0 ? "IN_VF"
                                                      : "OUTSIDE_VF")));
    fragment.Set("use_vfs", Any(ToInts(usage->use_vfs)));
    fragment.Set("cross_vf", Any(Bool(usage->cross_vf)));
    fragment.Set("escapes_vf", Any(Bool(usage->escapes_vf)));
    fragment.Set("uses_outside_vf", Any(Bool(usage->uses_outside_vf)));
    fragments.push_back(fragment);
  }
  if (fragments.empty())
    return table;
  table.Set("vf_index", Any(Integer(vf_index)));
  table.Set("fragments",
            Any(Array<Map<String, Any>>(fragments.begin(), fragments.end())));
  return table;
}

} // namespace vsimd
} // namespace tl
} // namespace tvm
