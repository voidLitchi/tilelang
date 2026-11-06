/*!
 * \file lift_vsimd.cc
 * \brief P1 pass: attach a typed VSimdRegion to every top-level T.Parallel loop.
 *
 * The pass is deliberately non-destructive: it never changes a Parallel body.
 * It records, per outer Parallel loop, the collapsed iteration space, the
 * ordered ops inside it, the captured context and a lift/opaque status, so P2
 * can plan without re-running scope analysis. Everything outside a top-level
 * Parallel loop is left untouched and stays with the legacy lowering.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "vsimd_capture_scope.h"
#include "vsimd_fragment.h"
#include "vsimd_ir.h"
#include "vsimd_memory_legality.h"
#include "vsimd_region.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;
using namespace vsimd;
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

namespace {

constexpr const char *kRegionKey = "tl.vsimd.region";

static void AddBufferUnique(std::vector<Buffer> *buffers, const Buffer &buffer) {
  for (const Buffer &existing : *buffers) {
    if (existing.same_as(buffer))
      return;
  }
  buffers->push_back(buffer);
}

/*
 * Fragment facts and fragment-only region grouping live in
 * `vsimd_fragment.{h,cc}`; the pass only calls into that module.
 */

/*! \brief Module-level facts shared by every VF of the function. */
struct ModuleFacts {
  std::vector<VSimdCaptureInfo> infos;
  std::vector<VSimdMemoryRegionStatus> memory_regions;
  /*! \brief Fragment facts; owned by the fragment module. */
  FragmentFacts fragments;
};

/*! \brief Per-VF state accumulated while its body is rewritten. */
struct VfState {
  int index{-1};
  std::vector<VSimdRegion> regions; // body order
};

/*!
 * \brief Attribute wrappers that do not introduce control flow or scope.
 *
 * They may wrap a top-level Parallel loop without changing what "top level"
 * means, so the walk looks through them.
 */
static bool IsTransparentAttr(const std::string &key) {
  return key == "tl.simdvf_scope" || key == "simdvf" || key == "tl.assume" ||
         key == tirx::attr::tilelang_assume || key == "tl.buffer_version" ||
         key == "tl.manual_multi_buffer";
}

/*! \brief Collapse the syntactic Parallel chain starting at `loop`. */
static std::vector<VSimdAxis> CollapseAxes(const ForNode *loop) {
  std::vector<VSimdAxis> axes;
  const ForNode *cur = loop;
  while (cur != nullptr) {
    axes.push_back(VSimdAxis(cur->loop_var, cur->min, cur->extent));
    const ForNode *next = cur->body.as<ForNode>();
    if (next == nullptr) {
      if (const auto *seq = cur->body.as<SeqStmtNode>()) {
        if (seq->seq.size() == 1)
          next = seq->seq[0].as<ForNode>();
      }
    }
    if (next == nullptr || next->kind != ForKind::kParallel)
      break;
    cur = next;
  }
  return axes;
}

/*!
 * \brief Conservative buffer footprint of a Parallel subtree.
 *
 * Used when the analysis could not reach the nest (manual VF, unsupported
 * statement): the region still has to tell P2 and the fragment grouping which
 * buffers it touched. Unknown-effect calls conservatively turn every seen
 * buffer into a write as well; a tileop call's destination may therefore appear
 * as a read, which is acceptable for an opaque region that P2 will not plan.
 */
class FootprintCollector : public StmtExprVisitor {
public:
  std::vector<Buffer> reads;
  std::vector<Buffer> writes;
  bool unknown_effects{false};

  void VisitExpr_(const BufferLoadNode *op) final {
    AddBufferUnique(&reads, op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    AddBufferUnique(&writes, op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (const auto *callee = op->op.as<OpNode>()) {
      const std::string name = callee->name;
      if (name == "tirx.call_extern" || name == "tirx.call_pure_extern" ||
          name.rfind("tl.simd.", 0) == 0 || name.rfind("tl.ascend_", 0) == 0) {
        unknown_effects = true;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

/*!
 * \brief Append a capture unless one for the same data var exists.
 *
 * Capture identity is the data var, not the Buffer object: a ``T.view`` shares
 * its base data var, so a footprint that observes both the view and the base
 * must not produce two captures (the verifier rejects duplicates).
 */
static void AddCaptureUnique(Array<VSimdCapture> *captures,
                             const VSimdCapture &capture) {
  for (const VSimdCapture &existing : *captures) {
    if (existing->buffer->data.same_as(capture->buffer->data))
      return;
  }
  captures->push_back(capture);
}

/*!
 * \brief Fill captures + an opaque op from a syntactic footprint.
 *
 * Capture kinds are "unknown" here: these buffers were not analysed, only
 * observed, so claiming a param/alloc/view role would be a guess.
 */
static void AttachSyntacticFootprint(const ForNode *loop,
                                     const std::string &reason,
                                     VSimdRegion &region) {
  FootprintCollector collector;
  collector(ffi::GetRef<Stmt>(loop));
  if (collector.unknown_effects) {
    for (const Buffer &buffer : collector.reads)
      AddBufferUnique(&collector.writes, buffer);
  }
  std::vector<Buffer> buffers = collector.reads;
  for (const Buffer &buffer : collector.writes)
    AddBufferUnique(&buffers, buffer);
  for (const Buffer &buffer : buffers) {
    AddCaptureUnique(&region->captures,
                     VSimdCapture(buffer, String("unknown"),
                                  StorageKindOf(buffer), ffi::Optional<Layout>()));
  }
  VSimdOp op(op_kind::kOpaque);
  op->status = String(region_status::kOpaque);
  op->reason = String(reason);
  op->reads = Array<Buffer>(collector.reads.begin(), collector.reads.end());
  op->writes = Array<Buffer>(collector.writes.begin(), collector.writes.end());
  region->ops.push_back(op);
}

/*!
 * \brief Build an OPAQUE region carrying the reason, axes and footprint.
 *
 * P2 (and the fragment grouping) needs to know which buffers an opaque region
 * touched, exactly as for a lifted one. When the analysis reached the nest the
 * captures/accesses come from it; otherwise they come from a syntactic
 * footprint of the Parallel subtree.
 */
static VSimdRegion MakeOpaqueRegion(const ForNode *loop, int vf_index,
                                    int region_index, const char *reason_code,
                                    const std::string &reason,
                                    const VSimdCaptureInfo *info) {
  VSimdRegion region;
  region->vf_index = vf_index;
  region->region_index = region_index;
  for (const VSimdAxis &axis : CollapseAxes(loop))
    region->axes.push_back(axis);
  region->status = String(region_status::kOpaque);
  region->reason_code = String(reason_code);
  region->reason = String(reason);
  if (info == nullptr) {
    AttachSyntacticFootprint(loop, reason, region);
    return region;
  }

  for (const auto &capture : info->captures) {
    ffi::Optional<Layout> layout;
    for (const auto &entry : info->layouts) {
      if (entry.first->data.same_as(capture.first->data))
        layout = entry.second;
    }
    AddCaptureUnique(&region->captures,
                     VSimdCapture(capture.first, String(capture.second),
                                  StorageKindOf(capture.first), layout));
  }
  VSimdOp op(op_kind::kOpaque);
  op->status = String(region_status::kOpaque);
  op->reason = String(reason);
  std::vector<Buffer> reads;
  std::vector<Buffer> writes;
  for (const AccessInfo &access : info->accesses) {
    AddBufferUnique(access.is_write ? &writes : &reads, access.buffer);
  }
  op->reads = Array<Buffer>(reads.begin(), reads.end());
  op->writes = Array<Buffer>(writes.begin(), writes.end());
  region->ops.push_back(op);
  return region;
}

static const VSimdCaptureInfo *FindCaptureInfo(const ModuleFacts &facts,
                                               int vf_index,
                                               const ForNode *root) {
  for (const VSimdCaptureInfo &info : facts.infos) {
    if (info.vf_index == vf_index && info.root_for == root)
      return &info;
  }
  return nullptr;
}

static const VSimdMemoryRegionStatus *FindMemoryStatus(const ModuleFacts &facts,
                                                       int vf_index,
                                                       const ForNode *root) {
  for (const VSimdMemoryRegionStatus &status : facts.memory_regions) {
    if (status.vf_index == vf_index && status.root == root)
      return &status;
  }
  return nullptr;
}

/*! \brief VF-level message from a capture entry that has no nest. */
static std::string VfMessage(const ModuleFacts &facts, int vf_index) {
  for (const VSimdCaptureInfo &info : facts.infos) {
    if (info.vf_index == vf_index && info.nest_index < 0)
      return info.message;
  }
  return "";
}

/*
 * The manual-SIMD / unsupported detection is VF-granular: P1 has no per-region
 * verdict for such a VF, and a region inside it may be perfectly liftable once
 * per-region analysis exists. Both therefore map to UNSUPPORTED, with the
 * VF-level message kept in the reason text; a dedicated MANUAL code would claim
 * something about the region that P1 did not check.
 */
static const char *ReasonCodeFor(const std::string &memory_status,
                                 bool internal_error) {
  if (internal_error)
    return reason_code::kInternal;
  if (memory_status == "ILLEGAL")
    return reason_code::kIllegal;
  if (memory_status == "UNPROVEN")
    return reason_code::kUnproven;
  return reason_code::kUnsupported;
}

/*! \brief Walk a VF body, annotating every top-level T.Parallel loop. */
class RegionBuilder {
public:
  RegionBuilder(const ModuleFacts &facts, VfState *state)
      : facts_(facts), state_(state) {}

  Stmt Build(const Stmt &stmt) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      bool changed = false;
      Array<Stmt> new_seq;
      for (const Stmt &child : seq->seq) {
        Stmt rewritten = Build(child);
        changed = changed || !rewritten.same_as(child);
        new_seq.push_back(rewritten);
      }
      if (!changed)
        return stmt;
      return SeqStmt(new_seq);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (!IsTransparentAttr(attr->attr_key))
        return stmt; // an opaque wrapper: do not reach inside
      Stmt rewritten = Build(attr->body);
      if (rewritten.same_as(attr->body))
        return stmt;
      auto node = ffi::make_object<AttrStmtNode>(*attr);
      node->body = rewritten;
      return AttrStmt(node);
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      if (loop->kind != ForKind::kParallel)
        return stmt; // serial loop: outside P1's unit
      return AttachRegion(loop);
    }
    return stmt;
  }

private:
  Stmt AttachRegion(const ForNode *loop) {
    const int region_index = static_cast<int>(state_->regions.size());
    VSimdRegion region;
    const VSimdCaptureInfo *info =
        FindCaptureInfo(facts_, state_->index, loop);
    const VSimdMemoryRegionStatus *memory_status =
        FindMemoryStatus(facts_, state_->index, loop);

    if (info == nullptr) {
      std::string reason = VfMessage(facts_, state_->index);
      // The manual / unsupported verdict is VF-granular: say so, because the
      // region itself was not analysed.
      reason = reason.empty()
                   ? "no capture info for this Parallel loop"
                   : "VF-level rejection: " + reason;
      region = MakeOpaqueRegion(loop, state_->index, region_index,
                                reason_code::kUnsupported, reason, nullptr);
    } else {
      const std::string memory =
          memory_status == nullptr ? "MISSING" : memory_status->status;
      VSimdRegionLift lift = BuildRegion(*info, memory);
      if (lift.lifted) {
        region = lift.region;
        region->vf_index = state_->index;
        region->region_index = region_index;
      } else {
        region = MakeOpaqueRegion(
            loop, state_->index, region_index,
            ReasonCodeFor(memory, lift.internal_error), lift.reason, info);
      }
    }

    state_->regions.push_back(region);
    For annotated = ffi::GetRef<For>(loop);
    annotated.CopyOnWrite()->annotations.Set(kRegionKey, Any(region));
    return annotated;
  }

  const ModuleFacts &facts_;
  VfState *state_;
};

class LiftMutator : public StmtExprMutator {
public:
  explicit LiftMutator(ModuleFacts facts) : facts_(std::move(facts)) {}

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF")
      return StmtExprMutator::VisitStmt_(op);
    const int vf_index = vf_index_++;
    SBlock block = ffi::GetRef<SBlock>(op);
    VfState state;
    state.index = vf_index;
    RegionBuilder builder(facts_, &state);
    Stmt body = builder.Build(block->body);
    // Fragment facts and fragment-only grouping are owned by the fragment
    // module: this is the only place the pass depends on it.
    GroupFragmentRegions(facts_.fragments, vf_index, &state.regions);
    Map<String, Any> fragments = FragmentTableForVf(facts_.fragments, vf_index);
    if (!body.same_as(block->body))
      block.CopyOnWrite()->body = body;
    if (!fragments.empty())
      block.CopyOnWrite()->annotations.Set(kFragmentAnnotationKey, fragments);
    return block;
  }

private:
  ModuleFacts facts_;
  int vf_index_{0};
};

class SimdVFCollector : public StmtExprVisitor {
public:
  bool found{false};
  bool has_parallel{false};
  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint == "SIMD_VF")
      found = true;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel)
      has_parallel = true;
    StmtExprVisitor::VisitStmt_(op);
  }
};

} // namespace

tvm::transform::Pass AscendLiftVSimd() {
  auto pass_func = [](PrimFunc f, const IRModule &mod, PassContext ctx) {
    if (!ctx->GetConfig<Bool>("tl.ascend_simd_lift", Bool(true)).value())
      return f;
    SimdVFCollector collector;
    collector(f->body);
    if (!collector.found || !collector.has_parallel)
      return f;

    ModuleFacts facts;
    facts.infos = AnalyzeVSimdCaptureInfo(f);
    facts.memory_regions = AnalyzeVSimdMemoryRegions(f);
    facts.fragments = CollectFragmentFacts(f);

    LiftMutator mutator(std::move(facts));
    PrimFuncNode *write_ptr = f.CopyOnWrite();
    write_ptr->body = mutator(std::move(write_ptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendLiftVSimd", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendLiftVSimd", []() {
    return AscendLiftVSimd();
  });
}

} // namespace tl
} // namespace tvm
