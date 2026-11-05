/*!
 * \file vsimd_ir.cc
 * \brief Typed, FFI-registered P1 vsimd region objects.
 */
#include "vsimd_ir.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt_functor.h>

#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {
namespace vsimd {

using namespace tirx;
using ffi::Array;
using ffi::Optional;
using ffi::String;

/* ------------------------------------------------------------------------- */
/* Object construction                                                       */
/* ------------------------------------------------------------------------- */

VSimdAxis::VSimdAxis(Var var, PrimExpr min, PrimExpr extent) {
  auto n = ffi::make_object<VSimdAxisNode>();
  n->var = std::move(var);
  n->min = std::move(min);
  n->extent = std::move(extent);
  data_ = std::move(n);
}

VSimdBufRegion::VSimdBufRegion(Buffer buffer, Array<PrimExpr> base,
                               Array<PrimExpr> extents) {
  auto n = ffi::make_object<VSimdBufRegionNode>();
  n->buffer = std::move(buffer);
  n->base = std::move(base);
  n->extents = std::move(extents);
  data_ = std::move(n);
}

VSimdAccess::VSimdAccess(Buffer buffer, Array<PrimExpr> indices,
                         Optional<PrimExpr> value, bool is_write,
                         Array<Integer> dep_axes, PrimExpr guard) {
  auto n = ffi::make_object<VSimdAccessNode>();
  n->buffer = std::move(buffer);
  n->indices = std::move(indices);
  n->value = std::move(value);
  n->is_write = is_write;
  n->dep_axes = std::move(dep_axes);
  n->guard = std::move(guard);
  data_ = std::move(n);
}

VSimdOp::VSimdOp(String kind) {
  auto n = ffi::make_object<VSimdOpNode>();
  n->kind = std::move(kind);
  n->status = String("OK");
  n->guard = TrueGuard();
  data_ = std::move(n);
}

VSimdOp AsOp(const ffi::ObjectRef &ref) {
  auto typed = ref.as<VSimdOp>();
  ICHECK(typed.has_value())
      << "expected a vsimd op reference, got " << ref.GetTypeKey();
  return typed.value();
}

Array<VSimdOp> ChildOps(const VSimdOp &op) {
  Array<VSimdOp> typed;
  for (const ffi::ObjectRef &child : op->ops)
    typed.push_back(AsOp(child));
  return typed;
}

bool VSimdOp::IsKind(const char *kind) const {
  return operator->()->kind == kind;
}

VSimdCapture::VSimdCapture(Buffer buffer, String kind, String storage,
                           Optional<Layout> layout) {
  auto n = ffi::make_object<VSimdCaptureNode>();
  n->buffer = std::move(buffer);
  n->kind = std::move(kind);
  n->storage = std::move(storage);
  n->layout = std::move(layout);
  data_ = std::move(n);
}

VSimdVarBinding::VSimdVarBinding(Var var, String kind,
                                 Optional<PrimExpr> value) {
  auto n = ffi::make_object<VSimdVarBindingNode>();
  n->var = std::move(var);
  n->kind = std::move(kind);
  n->value = std::move(value);
  data_ = std::move(n);
}

VSimdView::VSimdView(Buffer view, Buffer base) {
  auto n = ffi::make_object<VSimdViewNode>();
  n->view = std::move(view);
  n->base = std::move(base);
  data_ = std::move(n);
}

VSimdRegion::VSimdRegion() {
  auto n = ffi::make_object<VSimdRegionNode>();
  n->status = String(region_status::kLifted);
  data_ = std::move(n);
}

/* ------------------------------------------------------------------------- */
/* Small helpers                                                             */
/* ------------------------------------------------------------------------- */

String StorageKindOf(const Buffer &buffer) {
  if (!buffer.defined())
    return String("UNKNOWN");
  const std::string scope = buffer.scope();
  if (scope == "local.fragment")
    return String("FRAGMENT");
  if (scope.find("shared") != std::string::npos)
    return String("SHARED");
  if (scope == "global")
    return String("GLOBAL");
  return String("UNKNOWN");
}

PrimExpr TrueGuard() { return IntImm(DataType::Bool(), 1); }

namespace {

bool Fail(std::string *error, const std::string &message) {
  if (error != nullptr)
    *error = message;
  return false;
}

std::string ExprStr(const PrimExpr &expr) {
  std::ostringstream os;
  os << expr;
  return os.str();
}

std::string BufRegionStr(const VSimdBufRegionNode *region) {
  std::ostringstream os;
  os << region->buffer->name << "[";
  for (size_t i = 0; i < region->base.size(); ++i) {
    if (i)
      os << ", ";
    os << region->base[i];
  }
  os << "] x (";
  for (size_t i = 0; i < region->extents.size(); ++i) {
    if (i)
      os << ", ";
    os << region->extents[i];
  }
  os << ")";
  return os.str();
}

bool KindIsKnown(const String &kind) {
  return kind == op_kind::kParallel || kind == op_kind::kStore ||
         kind == op_kind::kReduce || kind == op_kind::kCopy ||
         kind == op_kind::kFill || kind == op_kind::kOpaque;
}

bool IsKnownReasonCode(const String &code) {
  return code == reason_code::kIllegal || code == reason_code::kUnproven ||
         code == reason_code::kUnsupported || code == reason_code::kInternal ||
         code == reason_code::kGroupOpaque;
}

bool CheckBufRegion(const Optional<VSimdBufRegion> &region,
                    const char *what, std::string *error) {
  if (!region.has_value() || !region.value().defined())
    return Fail(error, std::string("missing ") + what + " buffer region");
  const VSimdBufRegionNode *node = region.value().get();
  if (!node->buffer.defined())
    return Fail(error, std::string(what) + " buffer is undefined");
  if (node->base.size() != node->extents.size())
    return Fail(error, std::string(what) + " base/extent rank mismatch");
  if (node->base.size() != node->buffer->shape.size())
    return Fail(error, std::string(what) + " rank does not match the buffer");
  return true;
}

bool VerifyOp(const VSimdOp &op, int ctx_axes, std::string *error,
              std::unordered_set<const ffi::Object *> *used_buffers);

/*! \brief Collect the data vars of every buffer a store value/index reads. */
class ExprBufferCollector : public tirx::StmtExprVisitor {
public:
  explicit ExprBufferCollector(
      std::unordered_set<const ffi::Object *> *out)
      : out_(out) {}

  void VisitExpr_(const tirx::BufferLoadNode *op) final {
    out_->insert(op->buffer->data.get());
    tirx::StmtExprVisitor::VisitExpr_(op);
  }

private:
  std::unordered_set<const ffi::Object *> *out_;
};

void CollectExprBuffers(const PrimExpr &expr,
                        std::unordered_set<const ffi::Object *> *out) {
  if (!expr.defined())
    return;
  ExprBufferCollector collector(out);
  collector(expr);
}

bool VerifyStore(const VSimdOp &op, int ctx_axes, std::string *error,
                 std::unordered_set<const ffi::Object *> *used_buffers) {
  const VSimdOpNode *node = op.get();
  if (node->accesses.empty())
    return Fail(error, "store op has no access");
  int writes = 0;
  for (const VSimdAccess &access : node->accesses) {
    const VSimdAccessNode *acc = access.get();
    if (!acc->buffer.defined())
      return Fail(error, "store access has an undefined buffer");
    if (acc->indices.size() != acc->buffer->shape.size())
      return Fail(error, "store access rank does not match the buffer");
    if (!acc->guard.defined())
      return Fail(error, "store access has no guard");
    for (const Integer &axis : acc->dep_axes) {
      if (axis->value < 0 || axis->value >= ctx_axes)
        return Fail(error, "store access dep_axis out of range");
    }
    if (acc->is_write) {
      ++writes;
      if (!acc->value.has_value() || !acc->value.value().defined())
        return Fail(error, "store write access has no value");
    } else if (acc->value.has_value()) {
      return Fail(error, "load access must not carry a value");
    }
    used_buffers->insert(acc->buffer->data.get());
    // The value/indices may read buffers that are not listed as accesses.
    CollectExprBuffers(acc->value.value_or(PrimExpr()), used_buffers);
    for (const PrimExpr &index : acc->indices)
      CollectExprBuffers(index, used_buffers);
  }
  if (writes != 1)
    return Fail(error, "store op must contain exactly one write access");
  return true;
}

bool VerifyOp(const VSimdOp &op, int ctx_axes, std::string *error,
              std::unordered_set<const ffi::Object *> *used_buffers) {
  const VSimdOpNode *node = op.get();
  if (!KindIsKnown(node->kind))
    return Fail(error, "unknown op kind " + std::string(node->kind));
  if (!node->guard.defined())
    return Fail(error, "op has no guard");
  // Closed status vocabulary: only an opaque op may be OPAQUE.
  if (node->kind == op_kind::kOpaque) {
    if (node->status != String(region_status::kOpaque))
      return Fail(error, "opaque op must carry OPAQUE status");
  } else if (node->status != String("OK")) {
    return Fail(error, "non-opaque op must carry OK status");
  }

  if (node->kind == op_kind::kParallel) {
    if (node->axes.empty())
      return Fail(error, "parallel op has no axes");
    if (node->ops.empty())
      return Fail(error, "parallel op has no body");
    for (const ffi::ObjectRef &child : node->ops) {
      if (!VerifyOp(AsOp(child), ctx_axes + static_cast<int>(node->axes.size()),
                    error, used_buffers))
        return false;
    }
    return true;
  }
  if (node->kind == op_kind::kStore)
    return VerifyStore(op, ctx_axes, error, used_buffers);
  if (node->kind == op_kind::kReduce) {
    if (!node->source_op.has_value())
      return Fail(error, "reduce op has no source op name");
    if (!node->reduce_kind.has_value() || node->reduce_kind.value().empty())
      return Fail(error, "reduce op has no reduce kind");
    if (!node->reduce_dim.has_value() || !node->reduce_dim.value().defined())
      return Fail(error, "reduce op has no reduce dim");
    if (!node->clear.has_value() || !node->clear.value().defined())
      return Fail(error, "reduce op has no clear flag");
    if (!CheckBufRegion(node->src, "reduce src", error) ||
        !CheckBufRegion(node->dst, "reduce dst", error))
      return false;
    used_buffers->insert(node->src.value()->buffer->data.get());
    used_buffers->insert(node->dst.value()->buffer->data.get());
    return true;
  }
  if (node->kind == op_kind::kCopy) {
    if (!node->source_op.has_value())
      return Fail(error, "copy op has no source op name");
    if (!CheckBufRegion(node->src, "copy src", error) ||
        !CheckBufRegion(node->dst, "copy dst", error))
      return false;
    used_buffers->insert(node->src.value()->buffer->data.get());
    used_buffers->insert(node->dst.value()->buffer->data.get());
    return true;
  }
  if (node->kind == op_kind::kFill) {
    if (!node->source_op.has_value())
      return Fail(error, "fill op has no source op name");
    if (!node->fill_value.has_value() || !node->fill_value.value().defined())
      return Fail(error, "fill op has no value");
    if (!CheckBufRegion(node->dst, "fill dst", error))
      return false;
    used_buffers->insert(node->dst.value()->buffer->data.get());
    return true;
  }
  // opaque: its effect is not modelled, so it must explain itself and record a
  // conservative footprint whose buffers are captured like any other op's.
  if (node->reason.empty())
    return Fail(error, "opaque op has no reason");
  for (const Buffer &buffer : node->reads) {
    if (!buffer.defined())
      return Fail(error, "opaque op reads an undefined buffer");
    used_buffers->insert(buffer->data.get());
  }
  for (const Buffer &buffer : node->writes) {
    if (!buffer.defined())
      return Fail(error, "opaque op writes an undefined buffer");
    used_buffers->insert(buffer->data.get());
  }
  return true;
}

bool HasOpaqueOp(const Array<VSimdOp> &ops) {
  for (const VSimdOp &op : ops) {
    if (op->status == String(region_status::kOpaque))
      return true;
    if (op->kind == op_kind::kParallel && HasOpaqueOp(ChildOps(op)))
      return true;
  }
  return false;
}

} // namespace

/* ------------------------------------------------------------------------- */
/* Verification                                                              */
/* ------------------------------------------------------------------------- */

bool VerifyRegion(const VSimdRegion &region, std::string *error) {
  if (!region.defined())
    return Fail(error, "undefined region");
  const VSimdRegionNode *node = region.get();
  if (node->vf_index < 0)
    return Fail(error, "region has a negative vf_index");
  if (node->region_index < 0)
    return Fail(error, "region has a negative region_index");
  if (node->axes.empty())
    return Fail(error, "region has no Parallel axis");
  // Every region carries at least one op: a lifted region has its body, and an
  // opaque region carries the op that stands for what it could not model (the
  // P1 pass always emits one, including the syntactic-footprint fallback).
  if (node->ops.empty())
    return Fail(error, "region has no op");
  if (node->status != String(region_status::kLifted) &&
      node->status != String(region_status::kOpaque))
    return Fail(error, "region has an unknown status");
  if (node->status == String(region_status::kLifted)) {
    // A lifted region is liftable by itself: it cannot carry a reason.
    if (!node->reason_code.empty())
      return Fail(error, "LIFTED region must not carry a reason code");
    if (!node->reason.empty())
      return Fail(error, "LIFTED region must not carry a reason");
  } else {
    if (node->reason.empty())
      return Fail(error, "OPAQUE region has no reason");
    if (!IsKnownReasonCode(node->reason_code))
      return Fail(error, "OPAQUE region has an unknown reason code");
  }
  if (node->group < -1)
    return Fail(error, "region has an invalid group");

  std::unordered_set<const ffi::Object *> captured;
  for (const VSimdCapture &capture : node->captures) {
    const VSimdCaptureNode *cap = capture.get();
    if (!cap->buffer.defined())
      return Fail(error, "capture has an undefined buffer");
    if (cap->kind.empty())
      return Fail(error, "capture has no kind");
    if (cap->storage != "GLOBAL" && cap->storage != "SHARED" &&
        cap->storage != "FRAGMENT" && cap->storage != "UNKNOWN")
      return Fail(error, "capture has an invalid storage class");
    if (!captured.insert(cap->buffer->data.get()).second)
      return Fail(error, "duplicate capture of buffer " +
                             std::string(cap->buffer->name));
  }

  for (const VSimdView &view : node->local_views) {
    const VSimdViewNode *v = view.get();
    if (!v->view.defined() || !v->base.defined())
      return Fail(error, "local view is undefined");
    if (!v->view->data.same_as(v->base->data))
      return Fail(error, "local view resolves to a different data var");
  }

  // Captured variables and scalar definitions: a dump reads these fields
  // unconditionally, so their shape must be guaranteed here.
  for (const VSimdVarBinding &var : node->outer_vars) {
    if (!var->var.defined())
      return Fail(error, "outer var is undefined");
    if (var->kind.empty())
      return Fail(error, "outer var has no kind");
  }
  for (const VSimdVarBinding &binding : node->bindings) {
    if (!binding->var.defined())
      return Fail(error, "binding var is undefined");
    if (binding->kind != "bind")
      return Fail(error, "binding entry must have kind 'bind'");
    if (!binding->value.has_value() || !binding->value.value().defined())
      return Fail(error, "binding has no value");
  }
  for (const Var &var : node->shape_syms) {
    if (!var.defined())
      return Fail(error, "shape symbol is undefined");
  }
  for (const PrimExpr &cond : node->assumptions) {
    if (!cond.defined())
      return Fail(error, "assumption is undefined");
  }

  std::unordered_set<const ffi::Object *> used_buffers;
  for (const VSimdOp &op : node->ops) {
    if (!VerifyOp(op, static_cast<int>(node->axes.size()), error,
                  &used_buffers))
      return false;
  }
  for (const ffi::Object *buffer : used_buffers) {
    if (captured.count(buffer) == 0)
      return Fail(error, "region op uses an uncaptured buffer");
  }
  // A LIFTED region must not contain an unmodelled op: the two statuses would
  // contradict each other and P2 would consume a region it cannot plan.
  if (node->status == String(region_status::kLifted) && HasOpaqueOp(node->ops))
    return Fail(error, "LIFTED region contains an OPAQUE op");
  return true;
}

/* ------------------------------------------------------------------------- */
/* Dump                                                                      */
/* ------------------------------------------------------------------------- */

namespace {

void DumpAccesses(std::ostringstream &os, const VSimdOpNode *op,
                  const VSimdRegionNode *region, const std::string &pad) {
  // A T.view is a distinct Buffer sharing the base data var, so the layout must
  // be looked up by data var, not by Buffer identity.
  auto layout_of = [&](const Buffer &buffer) {
    for (const VSimdCapture &capture : region->captures) {
      if (capture->buffer->data.same_as(buffer->data))
        return capture->layout.has_value() ? "annotated" : "none";
    }
    return "none";
  };
  for (const VSimdAccess &access : op->accesses) {
    const VSimdAccessNode *acc = access.get();
    os << pad << (acc->is_write ? "W " : "R ") << acc->buffer->name << "[";
    for (size_t i = 0; i < acc->indices.size(); ++i) {
      if (i)
        os << ", ";
      os << acc->indices[i];
    }
    os << "]";
    if (acc->value.has_value() && acc->value.value().defined())
      os << " = " << acc->value.value();
    os << " dep={";
    for (size_t i = 0; i < acc->dep_axes.size(); ++i) {
      if (i)
        os << ",";
      os << acc->dep_axes[i]->value;
    }
    os << "} layout=" << layout_of(acc->buffer) << "\n";
  }
}

void DumpOp(std::ostringstream &os, const VSimdOpNode *op,
            const VSimdRegionNode *region, const std::string &pad) {
  if (op->kind == op_kind::kStore) {
    os << pad << "store\n";
    DumpAccesses(os, op, region, pad + "  ");
    return;
  }
  if (op->kind == op_kind::kParallel) {
    os << pad << "parallel";
    for (const VSimdAxis &axis : op->axes) {
      const VSimdAxisNode *a = axis.get();
      os << " " << a->var->name_hint << " in [" << a->min << ", "
         << (a->min + a->extent) << ")";
    }
    os << "\n";
    for (const ffi::ObjectRef &child : op->ops)
      DumpOp(os, AsOp(child).get(), region, pad + "  ");
    return;
  }
  if (op->kind == op_kind::kReduce) {
    os << pad << "reduce " << BufRegionStr(op->src.value().get()) << " -> "
       << BufRegionStr(op->dst.value().get())
       << " kind=" << op->reduce_kind.value()
       << " dim=" << op->reduce_dim.value()
       << " clear=" << op->clear.value() << "\n";
    return;
  }
  if (op->kind == op_kind::kCopy) {
    os << pad << "copy " << BufRegionStr(op->src.value().get()) << " -> "
       << BufRegionStr(op->dst.value().get());
    if (op->transpose.has_value() && op->transpose.value().defined())
      os << " transpose=" << op->transpose.value();
    os << "\n";
    return;
  }
  if (op->kind == op_kind::kFill) {
    os << pad << "fill " << BufRegionStr(op->dst.value().get())
       << " = " << op->fill_value.value() << "\n";
    return;
  }
  os << pad << "opaque reason=" << op->reason << "\n";
}

} // namespace

std::string DumpRegion(const VSimdRegion &region) {
  const VSimdRegionNode *node = region.get();
  std::ostringstream os;
  os << "region VF#" << node->vf_index << " region#" << node->region_index
     << " status=" << node->status << "\n";
  if (!node->reason.empty())
    os << "  reason: " << node->reason << "\n";
  os << "  axes:";
  for (const VSimdAxis &axis : node->axes) {
    const VSimdAxisNode *a = axis.get();
    os << " " << a->var->name_hint << " in [" << a->min << ", "
       << (a->min + a->extent) << ")";
  }
  os << "\n";
  os << "  captures:";
  for (const VSimdCapture &capture : node->captures)
    os << " " << capture->buffer->name << "[" << capture->kind << ","
       << capture->storage << "]";
  os << "\n";
  if (!node->outer_vars.empty()) {
    os << "  outer:";
    for (const VSimdVarBinding &var : node->outer_vars)
      os << " " << var->var->name_hint << "[" << var->kind << "]";
    os << "\n";
  }
  if (!node->shape_syms.empty()) {
    os << "  shape_syms:";
    for (const Var &var : node->shape_syms)
      os << " " << var->name_hint;
    os << "\n";
  }
  if (!node->local_views.empty()) {
    os << "  local_views:";
    for (const VSimdView &view : node->local_views)
      os << " " << view->view->name << " -> " << view->base->name;
    os << "\n";
  }
  {
    bool any = false;
    for (const VSimdCapture &capture : node->captures)
      any = any || capture->layout.has_value();
    if (any) {
      os << "  layouts:";
      for (const VSimdCapture &capture : node->captures) {
        if (capture->layout.has_value())
          os << " " << capture->buffer->name;
      }
      os << "\n";
    }
  }
  if (!node->assumptions.empty()) {
    os << "  assumptions:";
    for (const PrimExpr &cond : node->assumptions)
      os << " " << cond;
    os << "\n";
  }
  if (!node->bindings.empty()) {
    os << "  bindings:";
    for (const VSimdVarBinding &binding : node->bindings)
      os << " " << binding->var->name_hint << " = " << binding->value.value();
    os << "\n";
  }
  os << "  ops:\n";
  for (const VSimdOp &op : node->ops)
    DumpOp(os, op.get(), node, "    ");
  return os.str();
}

/* ------------------------------------------------------------------------- */
/* Reflection                                                                */
/* ------------------------------------------------------------------------- */

void VSimdAxisNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<VSimdAxisNode>()
      .def_ro("var", &VSimdAxisNode::var)
      .def_ro("min", &VSimdAxisNode::min)
      .def_ro("extent", &VSimdAxisNode::extent);
}

void VSimdBufRegionNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<VSimdBufRegionNode>()
      .def_ro("buffer", &VSimdBufRegionNode::buffer)
      .def_ro("base", &VSimdBufRegionNode::base)
      .def_ro("extents", &VSimdBufRegionNode::extents);
}

void VSimdAccessNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<VSimdAccessNode>()
      .def_ro("buffer", &VSimdAccessNode::buffer)
      .def_ro("indices", &VSimdAccessNode::indices)
      .def_ro("value", &VSimdAccessNode::value)
      .def_ro("is_write", &VSimdAccessNode::is_write)
      .def_ro("dep_axes", &VSimdAccessNode::dep_axes)
      .def_ro("guard", &VSimdAccessNode::guard);
}

void VSimdOpNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<VSimdOpNode>()
      .def_ro("kind", &VSimdOpNode::kind)
      .def_ro("axes", &VSimdOpNode::axes)
      .def_ro("ops", &VSimdOpNode::ops)
      .def_ro("accesses", &VSimdOpNode::accesses)
      .def_ro("src", &VSimdOpNode::src)
      .def_ro("dst", &VSimdOpNode::dst)
      .def_ro("source_op", &VSimdOpNode::source_op)
      .def_ro("reduce_kind", &VSimdOpNode::reduce_kind)
      .def_ro("reduce_dim", &VSimdOpNode::reduce_dim)
      .def_ro("clear", &VSimdOpNode::clear)
      .def_ro("fill_value", &VSimdOpNode::fill_value)
      .def_ro("transpose", &VSimdOpNode::transpose)
      .def_ro("guard", &VSimdOpNode::guard)
      .def_ro("status", &VSimdOpNode::status)
      .def_ro("reason", &VSimdOpNode::reason)
      .def_ro("reads", &VSimdOpNode::reads)
      .def_ro("writes", &VSimdOpNode::writes);
}

void VSimdCaptureNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<VSimdCaptureNode>()
      .def_ro("buffer", &VSimdCaptureNode::buffer)
      .def_ro("kind", &VSimdCaptureNode::kind)
      .def_ro("storage", &VSimdCaptureNode::storage)
      .def_ro("layout", &VSimdCaptureNode::layout);
}

void VSimdVarBindingNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<VSimdVarBindingNode>()
      .def_ro("var", &VSimdVarBindingNode::var)
      .def_ro("kind", &VSimdVarBindingNode::kind)
      .def_ro("value", &VSimdVarBindingNode::value);
}

void VSimdViewNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<VSimdViewNode>()
      .def_ro("view", &VSimdViewNode::view)
      .def_ro("base", &VSimdViewNode::base);
}

void VSimdRegionNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<VSimdRegionNode>()
      .def_ro("vf_index", &VSimdRegionNode::vf_index)
      .def_ro("region_index", &VSimdRegionNode::region_index)
      .def_ro("axes", &VSimdRegionNode::axes)
      .def_ro("ops", &VSimdRegionNode::ops)
      .def_ro("captures", &VSimdRegionNode::captures)
      .def_ro("outer_vars", &VSimdRegionNode::outer_vars)
      .def_ro("shape_syms", &VSimdRegionNode::shape_syms)
      .def_ro("local_views", &VSimdRegionNode::local_views)
      .def_ro("assumptions", &VSimdRegionNode::assumptions)
      .def_ro("bindings", &VSimdRegionNode::bindings)
      .def_ro("status", &VSimdRegionNode::status)
      .def_ro("reason_code", &VSimdRegionNode::reason_code)
      .def_ro("reason", &VSimdRegionNode::reason)
      .def_ro("group", &VSimdRegionNode::group);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  VSimdAxisNode::RegisterReflection();
  VSimdBufRegionNode::RegisterReflection();
  VSimdAccessNode::RegisterReflection();
  VSimdOpNode::RegisterReflection();
  VSimdCaptureNode::RegisterReflection();
  VSimdVarBindingNode::RegisterReflection();
  VSimdViewNode::RegisterReflection();
  VSimdRegionNode::RegisterReflection();
}

/* ------------------------------------------------------------------------- */
/* Object construction FFI                                                   */
/* ------------------------------------------------------------------------- */

/*
 * These builders are the programmatic entry point for the region IR. The P1
 * lift pass builds regions directly in C++, but the same objects must be
 * constructible from outside: the reference-expander/oracle tests need regions
 * for every op kind, and a future handwritten `T.vsimd` producer has to emit
 * the identical schema.
 */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  auto make_parallel = [](Array<VSimdAxis> axes, Array<VSimdOp> ops) {
    VSimdOp op(op_kind::kParallel);
    for (const VSimdAxis &axis : axes)
      op->axes.push_back(axis);
    for (const VSimdOp &child : ops)
      op->ops.push_back(ffi::ObjectRef(child));
    return op;
  };

  refl::GlobalDef()
      .def("tl.vsimd.make_axis",
           [](Var var, PrimExpr min, PrimExpr extent) {
             return VSimdAxis(var, min, extent);
           })
      .def("tl.vsimd.make_buf_region",
           [](Buffer buffer, Array<PrimExpr> base, Array<PrimExpr> extents) {
             return VSimdBufRegion(buffer, base, extents);
           })
      .def("tl.vsimd.make_access",
           [](Buffer buffer, Array<PrimExpr> indices, Optional<PrimExpr> value,
              bool is_write, Array<Integer> dep_axes,
              Optional<PrimExpr> guard) {
             return VSimdAccess(buffer, indices, value, is_write, dep_axes,
                                guard.value_or(TrueGuard()));
           })
      .def("tl.vsimd.make_capture",
           [](Buffer buffer, String kind, Optional<Layout> layout) {
             return VSimdCapture(buffer, kind, StorageKindOf(buffer), layout);
           })
      .def("tl.vsimd.make_view",
           [](Buffer view, Buffer base) { return VSimdView(view, base); })
      .def("tl.vsimd.make_var_binding",
           [](Var var, String kind, Optional<PrimExpr> value) {
             return VSimdVarBinding(var, kind, value);
           })
      .def("tl.vsimd.make_op_store",
           [](Array<VSimdAccess> accesses) {
             VSimdOp op(op_kind::kStore);
             op->accesses = std::move(accesses);
             return op;
           })
      .def("tl.vsimd.make_op_parallel", make_parallel)
      .def("tl.vsimd.make_op_reduce",
           [](VSimdBufRegion src, VSimdBufRegion dst, String source_op,
              String reduce_kind, PrimExpr dim, PrimExpr clear) {
             VSimdOp op(op_kind::kReduce);
             op->src = src;
             op->dst = dst;
             op->source_op = source_op;
             op->reduce_kind = reduce_kind;
             op->reduce_dim = dim;
             op->clear = clear;
             return op;
           })
      .def("tl.vsimd.make_op_copy",
           [](VSimdBufRegion src, VSimdBufRegion dst, String source_op,
              Optional<PrimExpr> transpose) {
             VSimdOp op(op_kind::kCopy);
             op->src = src;
             op->dst = dst;
             op->source_op = source_op;
             op->transpose = transpose;
             return op;
           })
      .def("tl.vsimd.make_op_fill",
           [](VSimdBufRegion dst, PrimExpr value, String source_op) {
             VSimdOp op(op_kind::kFill);
             op->dst = dst;
             op->fill_value = value;
             op->source_op = source_op;
             return op;
           })
      .def("tl.vsimd.make_op_opaque",
           [](String reason, Array<Buffer> reads, Array<Buffer> writes) {
             VSimdOp op(op_kind::kOpaque);
             op->status = String(region_status::kOpaque);
             op->reason = reason;
             op->reads = std::move(reads);
             op->writes = std::move(writes);
             return op;
           })
      .def("tl.vsimd.make_region",
           [](int vf_index, int region_index, Array<VSimdAxis> axes,
              Array<VSimdOp> ops, Array<VSimdCapture> captures,
              Array<VSimdVarBinding> outer_vars, Array<Var> shape_syms,
              Array<VSimdView> local_views, Array<PrimExpr> assumptions,
              Array<VSimdVarBinding> bindings, String status, String reason_code,
              String reason, int group) {
             VSimdRegion region;
             region->vf_index = vf_index;
             region->region_index = region_index;
             region->axes = std::move(axes);
             region->ops = std::move(ops);
             region->captures = std::move(captures);
             region->outer_vars = std::move(outer_vars);
             region->shape_syms = std::move(shape_syms);
             region->local_views = std::move(local_views);
             region->assumptions = std::move(assumptions);
             region->bindings = std::move(bindings);
             region->status = std::move(status);
             region->reason_code = std::move(reason_code);
             region->reason = std::move(reason);
             region->group = group;
             return region;
           })
      .def("tl.vsimd.verify_region",
           [](VSimdRegion region) {
             std::string error;
             if (!VerifyRegion(region, &error))
               return String(error);
             return String("");
           })
      // `DumpRegion` reads per-kind fields unconditionally and `DumpRegion` /
      // `ExpandVSimdRegionToLoops` assume a well-formed region, so both entry
      // points validate first instead of dereferencing a malformed object.
      .def("tl.vsimd.dump_region",
           [](VSimdRegion region) {
             std::string error;
             if (!VerifyRegion(region, &error))
               TVM_FFI_THROW(ValueError)
                   << "invalid vsimd region: " << error;
             return String(DumpRegion(region));
           });
}

} // namespace vsimd
} // namespace tl
} // namespace tvm
