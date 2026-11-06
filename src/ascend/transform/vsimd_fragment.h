/*!
 * \file vsimd_fragment.h
 * \brief P1 fragment facts and fragment-only region grouping.
 *
 * ``local.fragment`` is register-level storage. P1 records which fragments each
 * VF touches and how they are used across VFs, and groups regions that share a
 * fragment so they are lifted or dropped together. The physical realization
 * (register / spill / reject) stays with P2/P3; P1 only reports facts.
 *
 * This module owns the ``tl.vsimd.fragment`` annotation. Removing fragment
 * support means deleting this module, the include, the ``ModuleFacts`` field,
 * the three calls in the lift pass and that annotation key.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_VSIMD_FRAGMENT_H_
#define TVM_TL_ASCEND_TRANSFORM_VSIMD_FRAGMENT_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/tirx/function.h>

#include <string>
#include <vector>

#include "vsimd_ir.h"

namespace tvm {
namespace tl {
namespace vsimd {

/*! \brief Annotation key owned by this module (a VF SBlock annotation). */
inline constexpr const char *kFragmentAnnotationKey = "tl.vsimd.fragment";

/*! \brief The per-buffer facts P2 reads from the fragment table. */
struct FragmentUsage {
  tirx::Buffer buffer;
  /*! \brief VF that allocates the fragment, or -1 when defined outside a VF. */
  int definition_vf{-1};
  /*! \brief VFs that touch the fragment, ascending and deduplicated. */
  std::vector<int> use_vfs;
  /*! \brief The fragment is accessed outside every VF. */
  bool uses_outside_vf{false};
  /*! \brief Used by more than one VF, or by a VF other than its definition. */
  bool cross_vf{false};
  /*! \brief Defined in a VF and used outside it.
   *
   * The standard pipeline cannot produce this: the DSL rejects a VF-local
   * buffer used outside its defining region and ``VFChecker`` runs before P1,
   * so an IN_VF definition cannot escape. The flag is still computed from the
   * observed facts, which a handcrafted IR that skips the frontend can make
   * true.
   */
  bool escapes_vf{false};
};

/*! \brief Module-level fragment facts, collected once per function. */
struct FragmentFacts {
  /*! \brief Fragment buffers touched by each VF, in VF order. */
  std::vector<std::vector<tirx::Buffer>> vf_fragments;
  /*! \brief One entry per fragment buffer, in first-touch order. */
  std::vector<FragmentUsage> usages;
};

/*!
 * \brief Collect fragment definitions and uses for the whole function.
 *
 * Uses are counted anywhere in a VF (including top-level ``T.copy``/``T.reduce``
 * and serial loops), not only inside lifted regions. Only the outermost VF of a
 * nest is scanned, matching the P1 analysis order.
 */
TVM_DLL FragmentFacts CollectFragmentFacts(const tirx::PrimFunc &func);

/*!
 * \brief Group fragment-connected regions of one VF.
 *
 * Regions sharing a fragment get the same non-negative ``group`` id; if any
 * member is OPAQUE, the lifted members are downgraded to GROUP_OPAQUE while the
 * root cause keeps its own reason code. Only fragments form groups: shared and
 * global buffers always have the scalar fallback, so grouping them would make
 * P2 refuse more than necessary. The region objects are mutated in place.
 */
TVM_DLL void GroupFragmentRegions(const FragmentFacts &facts, int vf_index,
                                  std::vector<VSimdRegion> *regions);

/*!
 * \brief The ``tl.vsimd.fragment`` annotation payload for one VF.
 *
 * Returns an empty map when the VF touches no fragment. The map keys are the
 * P2-facing contract and are defined in this file only.
 */
TVM_DLL ffi::Map<ffi::String, ffi::Any>
FragmentTableForVf(const FragmentFacts &facts, int vf_index);

} // namespace vsimd
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VSIMD_FRAGMENT_H_
