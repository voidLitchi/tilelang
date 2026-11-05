/*!
 * \file vsimd_region.cc
 * \brief Build a VSimdRegion from an analyzed Parallel nest, plus the reference
 *        scalar expander used as a structural self-check.
 */
#include "vsimd_region.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vsimd_memory_legality.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace vsimd;
using ffi::Array;
using ffi::Optional;
using ffi::String;

namespace {

std::string ExprStr(const PrimExpr &expr) {
  std::ostringstream os;
  os << expr;
  return os.str();
}

bool ExprUsesVar(const PrimExpr &expr, const Var &var) {
  if (!expr.defined())
    return false;
  for (const Var &used : tirx::UndefinedVars(expr)) {
    if (used.same_as(var))
      return true;
  }
  return false;
}

std::vector<int> ComputeDepAxes(const AccessInfo &acc,
                                const std::vector<LoopInfo> &domain) {
  std::vector<int> dep;
  for (size_t axis = 0; axis < domain.size(); ++axis) {
    bool used = false;
    for (const PrimExpr &index : acc.indices) {
      if (ExprUsesVar(index, domain[axis].var)) {
        used = true;
        break;
      }
    }
    if (used)
      dep.push_back(static_cast<int>(axis));
  }
  return dep;
}

/*!
 * \brief Group the flat access list into store ops.
 *
 * The analysis reports accesses in source order with the store first, followed
 * by the loads used by its indices/value. One store statement therefore owns
 * one write access plus the loads that follow it until the next write.
 */
bool GroupStoreOps(const std::vector<AccessInfo> &accesses,
                   const std::vector<LoopInfo> &domain,
                   Array<VSimdOp> *ops) {
  std::vector<std::vector<VSimdAccess>> groups;
  for (const AccessInfo &acc : accesses) {
    std::vector<int> dep = ComputeDepAxes(acc, domain);
    Array<Integer> dep_axes;
    for (int axis : dep)
      dep_axes.push_back(Integer(axis));
    Optional<PrimExpr> value;
    if (acc.is_write && acc.value.defined())
      value = acc.value;
    VSimdAccess access(acc.buffer, acc.indices, value, acc.is_write, dep_axes,
                       TrueGuard());
    if (acc.is_write) {
      groups.emplace_back();
    } else if (groups.empty()) {
      // A load before any store cannot be attributed to a store statement.
      return false;
    }
    groups.back().push_back(std::move(access));
  }
  for (std::vector<VSimdAccess> &group : groups) {
    VSimdOp op(op_kind::kStore);
    op->accesses = Array<VSimdAccess>(group.begin(), group.end());
    ops->push_back(op);
  }
  return true;
}

Optional<Layout> FindLayout(const std::vector<std::pair<Buffer, Layout>> &layouts,
                            const Buffer &buffer) {
  for (const auto &entry : layouts) {
    if (entry.first->data.same_as(buffer->data))
      return entry.second;
  }
  return Optional<Layout>();
}

PrimExpr RegionCall(const VSimdBufRegion &region, int access_type) {
  std::vector<PrimExpr> args;
  args.push_back(BufferLoad(region->buffer, region->base));
  args.push_back(IntImm(DataType::Int(32), access_type));
  for (const PrimExpr &extent : region->extents)
    args.push_back(extent);
  return Call(DataType::Handle(), ::tvm::tl::region(), Array<PrimExpr>(args));
}

Stmt ExpandOp(const VSimdOp &op, std::string *unsupported) {
  const VSimdOpNode *node = op.get();
  if (node->kind == op_kind::kStore) {
    std::vector<Stmt> stores;
    for (const VSimdAccess &access : node->accesses) {
      if (!access->is_write)
        continue;
      stores.push_back(
          BufferStore(access->buffer, access->value.value(), access->indices));
    }
    if (stores.empty())
      return Evaluate(0);
    return stores.size() == 1 ? stores[0] : SeqStmt(stores);
  }
  if (node->kind == op_kind::kParallel) {
    std::vector<Stmt> body;
    for (const ffi::ObjectRef &child : node->ops) {
      Stmt expanded = ExpandOp(AsOp(child), unsupported);
      if (!expanded.defined())
        return Stmt();
      body.push_back(expanded);
    }
    Stmt stmt = body.size() == 1 ? body[0] : SeqStmt(body);
    for (auto it = node->axes.rbegin(); it != node->axes.rend(); ++it) {
      stmt = For((*it)->var, (*it)->min, (*it)->extent, ForKind::kSerial, stmt);
    }
    return stmt;
  }
  // A copy transpose is not modelled yet. Refuse instead of emitting a copy
  // that silently drops it: the reference expander is a structural oracle, so
  // losing a semantic here would make the oracle unsound.
  if (node->kind == op_kind::kCopy && node->transpose.has_value() &&
      node->transpose.value().defined()) {
    if (unsupported != nullptr) {
      *unsupported =
          "reference expander does not support a transposed copy yet "
          "(transpose=" +
          ExprStr(node->transpose.value()) + ")";
    }
    return Stmt();
  }
  // An opaque op has no modelled effect, so it cannot be reconstructed at all.
  if (node->kind == op_kind::kOpaque) {
    if (unsupported != nullptr) {
      *unsupported = "reference expander cannot represent an opaque op (" +
                     std::string(node->reason) + ")";
    }
    return Stmt();
  }
  // Structured ops are re-emitted as the tileop call they came from; the
  // reference expander is a structural oracle, not a scalar interpreter.
  if (node->source_op.has_value()) {
    std::vector<PrimExpr> args;
    if (node->kind == op_kind::kReduce) {
      args.push_back(RegionCall(node->src.value(), 1));
      args.push_back(RegionCall(node->dst.value(), 2));
      args.push_back(StringImm(node->reduce_kind.value()));
      args.push_back(node->reduce_dim.value());
      args.push_back(node->clear.value());
    } else if (node->kind == op_kind::kCopy) {
      args.push_back(RegionCall(node->src.value(), 1));
      args.push_back(RegionCall(node->dst.value(), 2));
    } else if (node->kind == op_kind::kFill) {
      args.push_back(RegionCall(node->dst.value(), 2));
      args.push_back(node->fill_value.value());
    }
    return Evaluate(Call(DataType::Handle(), Op::Get(node->source_op.value()),
                         Array<PrimExpr>(args)));
  }
  // No builder should produce an op the expander cannot reconstruct; refuse
  // loudly rather than dropping it (silently losing an op would make the
  // structural oracle unsound).
  if (unsupported != nullptr) {
    *unsupported = "reference expander cannot represent op kind " +
                   std::string(node->kind) + " (missing source_op)";
  }
  return Stmt();
}

void CollectExpectedAxes(const Array<VSimdOp> &ops,
                         std::vector<const VSimdAxisNode *> *axes) {
  for (const VSimdOp &op : ops) {
    if (op->kind != op_kind::kParallel)
      continue;
    for (const VSimdAxis &axis : op->axes)
      axes->push_back(axis.get());
    CollectExpectedAxes(ChildOps(op), axes);
  }
}

int CountWrites(const Array<VSimdOp> &ops) {
  int writes = 0;
  for (const VSimdOp &op : ops) {
    if (op->kind == op_kind::kStore) {
      for (const VSimdAccess &access : op->accesses)
        writes += access->is_write ? 1 : 0;
    } else if (op->kind == op_kind::kParallel) {
      writes += CountWrites(ChildOps(op));
    }
  }
  return writes;
}

} // namespace

VSimdRegionLift BuildRegion(const VSimdCaptureInfo &info,
                            const std::string &memory_status) {
  VSimdRegionLift result;
  if (info.status != "CLOSED") {
    result.reason = "capture/scope is not CLOSED";
    return result;
  }
  if (memory_status != "LEGAL") {
    result.reason = "memory legality is " + memory_status;
    return result;
  }
  if (info.domain.empty()) {
    result.reason = "region requires at least one T.Parallel axis";
    return result;
  }

  bool has_write = false;
  for (const AccessInfo &acc : info.accesses) {
    if (acc.buffer.scope() == "local.var") {
      result.reason = "local.var access is not liftable";
      return result;
    }
    if (acc.buffer->shape.size() != acc.indices.size()) {
      result.reason = "access rank does not match buffer rank";
      return result;
    }
    has_write = has_write || acc.is_write;
  }
  if (!has_write) {
    result.reason = "nest has no BufferStore";
    return result;
  }

  VSimdRegion region;
  region->vf_index = info.vf_index;
  region->region_index = info.nest_index;
  for (const LoopInfo &loop : info.domain)
    region->axes.push_back(VSimdAxis(loop.var, loop.min, loop.extent));
  if (!GroupStoreOps(info.accesses, info.domain, &region->ops)) {
    result.reason = "access list does not start with a BufferStore";
    return result;
  }
  for (const auto &capture : info.captures) {
    region->captures.push_back(
        VSimdCapture(capture.first, String(capture.second),
                     StorageKindOf(capture.first),
                     FindLayout(info.layouts, capture.first)));
  }
  for (const auto &var : info.outer_vars)
    region->outer_vars.push_back(
        VSimdVarBinding(var.first, String(var.second), Optional<PrimExpr>()));
  for (const Var &var : info.shape_syms)
    region->shape_syms.push_back(var);
  for (const auto &view : info.local_views)
    region->local_views.push_back(VSimdView(view.first, view.second));
  for (const PrimExpr &cond : info.assumptions)
    region->assumptions.push_back(cond);
  for (const auto &binding : info.bindings)
    region->bindings.push_back(VSimdVarBinding(binding.first, String("bind"),
                                               Optional<PrimExpr>(binding.second)));
  region->status = String(region_status::kLifted);
  region->group = -1;

  std::string verify_error;
  if (!VerifyRegion(region, &verify_error)) {
    LOG(WARNING) << "[vsimd] internal region verification failed (compiler "
                    "bug): "
                 << verify_error;
    result.internal_error = true;
    result.reason = "internal region verification failed: " + verify_error;
    return result;
  }

  std::string unsupported;
  Stmt expanded = ExpandVSimdRegionToLoops(region, &unsupported);
  if (!unsupported.empty() || !expanded.defined()) {
    LOG(WARNING) << "[vsimd] internal region expansion failed (compiler bug): "
                 << unsupported;
    result.internal_error = true;
    result.reason = "internal expander failure: " + unsupported;
    return result;
  }
  VSimdRegionVerify verify = VerifyExpandedRegion(region, expanded);
  if (!verify.ok) {
    LOG(WARNING) << "[vsimd] internal region expansion verification failed "
                    "(compiler bug): "
                 << verify.reason;
    result.internal_error = true;
    result.reason = "internal expander verification failed: " + verify.reason;
    return result;
  }

  result.lifted = true;
  result.region = std::move(region);
  return result;
}

Stmt ExpandVSimdRegionToLoops(const VSimdRegion &region,
                              std::string *unsupported) {
  std::vector<Stmt> body;
  for (const VSimdOp &op : region->ops) {
    Stmt expanded = ExpandOp(op, unsupported);
    if (!expanded.defined())
      return Stmt();
    body.push_back(expanded);
  }
  Stmt result = body.size() == 1 ? body[0] : SeqStmt(body);
  // Reference expansion is intentionally serial scalar loops; it is a
  // structural oracle, not the P3 physical lowering.
  for (auto it = region->axes.rbegin(); it != region->axes.rend(); ++it) {
    result = For((*it)->var, (*it)->min, (*it)->extent, ForKind::kSerial, result);
  }
  // T.view aliases are DeclBuffer definitions, not allocations. They must be
  // declared before the reconstructed loop or the expanded IR references an
  // undeclared view buffer.
  for (auto it = region->local_views.rbegin(); it != region->local_views.rend();
       ++it) {
    result = SeqStmt({DeclBuffer((*it)->view), result});
  }
  return result;
}

VSimdRegionVerify VerifyExpandedRegion(const VSimdRegion &region,
                                       const Stmt &expanded) {
  using BufferSet =
      std::unordered_set<Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
  BufferSet visible;
  for (const VSimdCapture &capture : region->captures)
    visible.insert(capture->buffer);
  for (const VSimdView &view : region->local_views)
    visible.insert(view->view);

  class VerifyVisitor : public StmtExprVisitor {
  public:
    BufferSet declared;
    std::vector<const BufferStoreNode *> stores;
    std::vector<const ForNode *> loops;
    std::vector<Buffer> loads;
    std::string error;

    void VisitStmt_(const DeclBufferNode *op) final {
      declared.insert(op->buffer);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const ForNode *op) final {
      loops.push_back(op);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      stores.push_back(op);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const BufferLoadNode *op) final {
      loads.push_back(op->buffer);
      StmtExprVisitor::VisitExpr_(op);
    }
  } verify;

  verify(expanded);

  VSimdRegionVerify result;
  for (const VSimdView &view : region->local_views) {
    if (!verify.declared.count(view->view)) {
      result.ok = false;
      result.reason =
          "local view is not declared: " + std::string(view->view->name);
      return result;
    }
  }

  auto visible_buffer = [&](const Buffer &buffer) {
    return visible.count(buffer) || verify.declared.count(buffer);
  };
  for (const Buffer &buffer : verify.loads) {
    if (!visible_buffer(buffer)) {
      result.ok = false;
      result.reason = "load from undeclared buffer: " +
                      std::string(buffer->name);
      return result;
    }
  }

  std::vector<const VSimdAxisNode *> expected;
  for (const VSimdAxis &axis : region->axes)
    expected.push_back(axis.get());
  CollectExpectedAxes(region->ops, &expected);
  if (verify.loops.size() != expected.size()) {
    result.ok = false;
    result.reason = "expanded loop count does not match region axes";
    return result;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    const ForNode *loop = verify.loops[i];
    if (!loop->loop_var.same_as(expected[i]->var) ||
        !ffi::StructuralEqual()(loop->min, expected[i]->min) ||
        !ffi::StructuralEqual()(loop->extent, expected[i]->extent)) {
      result.ok = false;
      result.reason = "expanded domain does not match region axes";
      return result;
    }
  }

  const int writes = CountWrites(region->ops);
  if (static_cast<int>(verify.stores.size()) != writes) {
    result.ok = false;
    result.reason = "expanded store count does not match region stores";
    return result;
  }
  std::vector<const VSimdAccessNode *> expected_stores;
  for (const VSimdOp &op : region->ops) {
    if (op->kind != op_kind::kStore)
      continue;
    for (const VSimdAccess &access : op->accesses) {
      if (access->is_write)
        expected_stores.push_back(access.get());
    }
  }
  for (size_t i = 0; i < expected_stores.size(); ++i) {
    const BufferStoreNode *store = verify.stores[i];
    const VSimdAccessNode *acc = expected_stores[i];
    if (!store->buffer.same_as(acc->buffer) ||
        !ffi::StructuralEqual()(store->indices, acc->indices) ||
        !ffi::StructuralEqual()(store->value, acc->value.value())) {
      result.ok = false;
      result.reason = "expanded store does not match region access";
      return result;
    }
  }
  return result;
}

String AnalyzeVSimdRegionReport(const PrimFunc &func) {
  std::vector<VSimdMemoryRegionStatus> memory_regions =
      AnalyzeVSimdMemoryRegions(func);
  std::vector<VSimdCaptureInfo> infos = AnalyzeVSimdCaptureInfo(func);

  std::ostringstream out;
  for (const VSimdCaptureInfo &info : infos) {
    if (info.nest_index < 0) {
      out << "VF#" << info.vf_index << " status=OPAQUE";
      if (!info.message.empty())
        out << " reason=" << info.message;
      out << "\n";
      continue;
    }
    // Gate each nest on its own legality status: an illegal sibling nest must
    // not disable a legal one (the P1 pass already works per nest).
    std::string memory_status = "MISSING";
    for (const VSimdMemoryRegionStatus &status : memory_regions) {
      if (status.vf_index == info.vf_index &&
          status.nest_index == info.nest_index) {
        memory_status = status.status;
        break;
      }
    }
    VSimdRegionLift lift = BuildRegion(info, memory_status);
    out << "VF#" << info.vf_index << " nest " << info.nest_index << ": ";
    if (!lift.lifted) {
      if (lift.internal_error) {
        out << "status=INTERNAL_ERROR reason=" << lift.reason << "\n";
      } else {
        out << "status=FALLBACK reason=" << lift.reason << "\n";
      }
      continue;
    }
    Stmt expanded_stmt = ExpandVSimdRegionToLoops(lift.region);
    if (!expanded_stmt.defined()) {
      out << "status=INTERNAL_ERROR reason=reference expander cannot represent "
             "this region\n";
      continue;
    }
    out << "status=LIFTED\n";
    out << DumpRegion(lift.region);
    std::ostringstream expanded;
    expanded << expanded_stmt;
    out << "  expanded:\n    " << expanded.str() << "\n";
    out << "  verify: OK\n";
  }
  if (out.str().empty())
    return String("no SIMD_VF blocks\n");
  return String(out.str());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.vsimd.expand_region", [](VSimdRegion region) {
        // The expander reads per-kind fields unconditionally, so validate the
        // region at the entry point instead of dereferencing a malformed one.
        std::string error;
        if (!VerifyRegion(region, &error))
          TVM_FFI_THROW(ValueError) << "invalid vsimd region: " << error;
        std::string unsupported;
        Stmt expanded = ExpandVSimdRegionToLoops(region, &unsupported);
        Array<ffi::Any> out;
        out.push_back(expanded.defined() ? ffi::Any(expanded) : ffi::Any());
        out.push_back(ffi::Any(String(unsupported)));
        return out;
      });
}

} // namespace tl
} // namespace tvm
