// Explicit mixed-core schedule from the qsa-old AIV kernel (919dd815),
// M=16, 2176 slots, three pipeline stages. Keep the index experiment
// compile-time so the control retains the original instructions.
#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/nd2nz_copy.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void main_kernel_simd_vf_3(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_bool full = asc_create_mask_b32(PAT_ALL);
  vector_bool b16 = asc_create_mask_b16(PAT_VL128);
  auto lv = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[27584])), 0);
  auto ones = simd_inst::vadds(simd_inst::vmuls(lv, float(0x0p+0f/*0.000000e+00*/), full, MODE_ZEROING), float(0x1p+0f/*1.000000e+00*/), full, MODE_ZEROING);
  auto guard = simd_inst::vcmps_le(lv, float(0x0p+0f/*0.000000e+00*/), full);
  auto denom = simd_inst::vsel(ones, lv, guard);
  auto recip = simd_inst::vdiv(ones, denom, full, MODE_ZEROING);
  simd_inst::vsts_norm(recip, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[27584])), 0, full);
  simd_inst::mem_bar(VST_VLD);
  for (int32_t r = 0; r < 8; ++r) {
    auto rp = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(r + 27584)])), 0);
    for (int32_t _tmp = 0; _tmp < 2; ++_tmp) {
      auto v = simd_inst::vld_x2<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((r * 256) + (_tmp * 128)) + 22784)])), DINTLV_B32);
      auto f0 = simd_inst::vmul(v.v0, rp, full, MODE_ZEROING);
      auto f1 = simd_inst::vmul(v.v1, rp, full, MODE_ZEROING);
      auto c0 = simd_inst::vcvt<bfloat16_t>(f0, full, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
      auto c1 = simd_inst::vcvt<bfloat16_t>(f1, full, ROUND_R, RS_DISABLE, PART_ODD, MODE_ZEROING);
        vector_u16 v_ = simd_inst::vor((*(vector_u16 *)(&(c0))), (*(vector_u16 *)(&(c1))), b16, MODE_ZEROING);
      simd_inst::vsts_norm((*(vector_bf16 *)(&(v_))), (__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((r * 256) + (_tmp * 128)) + 51968)])), 0, b16);
    }
  }
}

__simd_vf__ inline void main_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_bool full = asc_create_mask_b32(PAT_ALL);
  vector_bool v = asc_create_mask_b32(PAT_ALL);
  auto zero = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING);
  simd_inst::vsts_norm(simd_inst::vdup<float>(float(-0x1p+30f/*-1.073742e+09*/), v, MODE_ZEROING), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[27520])), 0, full);
  simd_inst::vsts_norm(zero, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[27584])), 0, full);
  for (int32_t r = 0; r < 8; ++r) {
    for (int32_t _tmp = 0; _tmp < 4; ++_tmp) {
      simd_inst::vsts_norm(zero, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((r * 256) + (_tmp * 64)) + 22784)])), 0, full);
    }
  }
}

__simd_vf__ inline void main_kernel_simd_vf_1(__ubuf__ uint8_t* buf_dyn_shmem, int32_t task_block_1) {
  __ubuf__ void* p_nz_ptr = nullptr;
  vector_bool full = asc_create_mask_b32(PAT_ALL);
  vector_bool one = asc_create_mask_b32(PAT_VL1);
  vector_bool b16 = asc_create_mask_b16(PAT_VL128);
  auto v_lo = simd_inst::vcvt<float>(simd_inst::vlds_norm<int32_t>((__ubuf__ int32_t*)(&(((__ubuf__ int32_t*)buf_dyn_shmem)[((((task_block_1 + 1) & 1) * 128) + 27264)])), 0), full, ROUND_R, MODE_ZEROING);
  auto v_hi = simd_inst::vcvt<float>(simd_inst::vlds_norm<int32_t>((__ubuf__ int32_t*)(&(((__ubuf__ int32_t*)buf_dyn_shmem)[((((task_block_1 + 1) & 1) * 128) + 27328)])), 0), full, ROUND_R, MODE_ZEROING);
  auto pen_lo = simd_inst::vmuls(simd_inst::vadds(v_lo, float(-0x1p+0f/*-1.000000e+00*/), full, MODE_ZEROING), float(0x1.93e5939a08ceap+99f/*1.000000e+30*/), full, MODE_ZEROING);
  auto pen_hi = simd_inst::vmuls(simd_inst::vadds(v_hi, float(-0x1p+0f/*-1.000000e+00*/), full, MODE_ZEROING), float(0x1.93e5939a08ceap+99f/*1.000000e+30*/), full, MODE_ZEROING);
  p_nz_ptr = (&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((((task_block_1 + 1) & 1) * 1152) + 49664)]));
  for (int32_t r = 0; r < 8; ++r) {
    auto s0 = simd_inst::vmuls(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((((task_block_1 + 1) & 1) * 1024) + (r * 128)) + 20736)])), 0), float(0x1p-4f/*6.250000e-02*/), full, MODE_ZEROING);
    auto s1 = simd_inst::vmuls(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((((task_block_1 + 1) & 1) * 1024) + (r * 128)) + 20800)])), 0), float(0x1p-4f/*6.250000e-02*/), full, MODE_ZEROING);
    auto s0_1 = simd_inst::vadd(s0, pen_lo, full, MODE_ZEROING);
    auto s1_1 = simd_inst::vadd(s1, pen_hi, full, MODE_ZEROING);
    simd_inst::vsts_norm(s0_1, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((((task_block_1 + 1) & 1) * 1024) + (r * 128)) + 20736)])), 0, full);
    simd_inst::vsts_norm(s1_1, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((((task_block_1 + 1) & 1) * 1024) + (r * 128)) + 20800)])), 0, full);
    auto m_all = simd_inst::vmax(s0_1, s1_1, full, MODE_ZEROING);
    auto old = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(r + 27520)])), 0);
    simd_inst::vsts_1st(old, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(r + 27712)])), 0, one);
    auto m_all_1 = simd_inst::vmax(m_all, old, full, MODE_ZEROING);
    auto mrow = simd_inst::vcmax(m_all_1, full, MODE_ZEROING);
    simd_inst::vsts_1st(mrow, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(r + 27520)])), 0, one);
  }
  simd_inst::mem_bar(VST_VLD);
  for (int32_t r_1 = 0; r_1 < 8; ++r_1) {
    auto mrow_1 = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(r_1 + 27520)])), 0);
    auto v = simd_inst::vld_x2<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((((task_block_1 + 1) & 1) * 1024) + (r_1 * 128)) + 20736)])), DINTLV_B32);
    auto p0 = simd_inst::vexpdif(v.v0, mrow_1, full);
    auto p1 = simd_inst::vexpdif(v.v1, mrow_1, full);
    auto sumall = simd_inst::vadd(p0, p1, full, MODE_ZEROING);
    auto b0 = simd_inst::vcvt<bfloat16_t>(p0, full, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
    auto b1 = simd_inst::vcvt<bfloat16_t>(p1, full, ROUND_R, RS_DISABLE, PART_ODD, MODE_ZEROING);
    auto packed = simd_inst::vor((*(vector_u16 *)(&(b0))), (*(vector_u16 *)(&(b1))), b16, MODE_ZEROING);
    p_nz_ptr = simd_inst::vsstb((*(vector_bf16 *)(&(packed))), (__ubuf__ bfloat16_t*)p_nz_ptr, 589825, b16, POST_UPDATE);
    auto psum = simd_inst::vcadd(sumall, full, MODE_ZEROING);
    auto alpha = simd_inst::vexpdif(simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(r_1 + 27712)])), 0), mrow_1, full);
    simd_inst::vsts_1st(alpha, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(r_1 + 27648)])), 0, one);
    auto lnew = simd_inst::vadd(simd_inst::vmul(simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(r_1 + 27584)])), 0), alpha, full, MODE_ZEROING), psum, one, MODE_ZEROING);
    simd_inst::vsts_1st(lnew, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(r_1 + 27584)])), 0, one);
  }
}

__simd_vf__ inline void main_kernel_simd_vf_2(__ubuf__ uint8_t* buf_dyn_shmem, int32_t task_block_1) {
  vector_bool full = asc_create_mask_b32(PAT_ALL);
  for (int32_t r = 0; r < 8; ++r) {
    auto a = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(r + 27648)])), 0);
    for (int32_t _tmp = 0; _tmp < 4; ++_tmp) {
      auto o = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((r * 256) + (_tmp * 64)) + 22784)])), 0);
      auto t = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((((task_block_1 + 1) & 1) * 2048) + (r * 256)) + (_tmp * 64)) + 16640)])), 0);
      simd_inst::vsts_norm(simd_inst::vadd(simd_inst::vmul(o, a, full, MODE_ZEROING), t, full, MODE_ZEROING), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((r * 256) + (_tmp * 64)) + 22784)])), 0, full);
    }
  }
}

// Each time step addresses a logical tile. The event rings advance across
// task boundaries; only Q, the online reductions and the output are per task.
template <int CoreCount>
__aicore__ inline int qsa_task_for_tile(int tile) {
  return (int)block_idx + (tile / 17) * CoreCount;
}

template <int CoreCount>
__aicore__ inline int qsa_row_slot_offset(int tile) {
  if constexpr (CoreCount == 0) return tile * 128;
  return ((tile / 17) & 1) * 2176 + (tile % 17) * 128;
}

// Every pair starts on an even ND version. The experiment issues both
// gathers before scattering and draining the two slots in their old order.
#ifdef QSA_PAIR_CHUNKS
#define QSA_GATHER_SLOT(chunk) (chunk)
#define QSA_GATHER_QUADS (issue < 2 ? 8 : 0)
#else
#define QSA_GATHER_SLOT(chunk) (KV_ub_version_counter_1 & 1)
#define QSA_GATHER_QUADS 8
#endif
// In the relay schedule K and V share three L1 slots. The hybrid uses two
// private K slots after P, and the first two original slots for V.
#ifdef QSA_K_AHEAD
#define QSA_K_OFFSET(t) (qsa_k_l1_base + (((t) - 33) & 1) * qsa_k_tile_size)
#define QSA_V_READ_SLOT(t) (((t) - 34) & 1)
#define QSA_V_WRITE_SLOT(t) (((t) - 2) & 1)
#else
#define QSA_K_OFFSET(t) (((t) % 3) * 32768)
#define QSA_V_READ_SLOT(t) (((t) + 2) % 3)
#define QSA_V_WRITE_SLOT(t) (((t) + 1) % 3)
#endif
template <bool Prefetch, int LastFullQuad, int CoreCount, int TaskCount>
__aicore__ inline void qsa_old_task(__ubuf__ uint8_t* buf_dyn_shmem, __gm__ int32_t* Indices, __gm__ bfloat16_t* K, __gm__ bfloat16_t* O, __gm__ bfloat16_t* Q, __gm__ bfloat16_t* V, __gm__ int32_t* Valid, int runtime_last_full_quad, int runtime_task_count) {
  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __cc__ uint8_t *buf_dyn_l0c = (__cc__ uint8_t *)0;
  __cb__ uint8_t *buf_dyn_l0b = (__cb__ uint8_t *)0;
  __ca__ uint8_t *buf_dyn_l0a = (__ca__ uint8_t *)0;
  // Dynamic callers pass the packed task count and KV boundary. Static
  // callers use the template constants, which the compiler can fold away.
  int last_full_quad = LastFullQuad;
  int task_count = TaskCount;
  if constexpr (LastFullQuad < 0) {
    last_full_quad = runtime_last_full_quad;
  }
  if constexpr (TaskCount < 0) {
    task_count = runtime_task_count;
  }
  int tiles = 17;
  if constexpr (CoreCount != 0) {
    if ((int)block_idx >= task_count) return;
    tiles *= (task_count - 1 - (int)block_idx) / CoreCount + 1;
  }
  int32_t KV_ub_version_counter_1 = 0;
  if ASC_IS_AIC {
    asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
    asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
#ifdef QSA_K_AHEAD
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
#endif
    for (int32_t task_block = 0; task_block < tiles + 34; ++task_block) {
      if ((task_block < tiles + 33) && (33 <= task_block)) {
        asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
        if (CoreCount ? (task_block - 33) % 17 == 0 : task_block == 33) {
          asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(16), 0);
          asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 98304), (&(Q[qsa_task_for_tile<CoreCount>(task_block - 33) * 4096])), 512, static_cast<asc_load_l2_cache_mode>(4), 16, 256, 0, 0);
        }
        asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
#ifdef QSA_K_AHEAD
        int k_tile = task_block - 33;
        if (!QSA_K_AHEAD || k_tile == 0) {
          qsa_load_k_tile<CoreCount>(k_tile, Indices, K, (__cbuf__ bfloat16_t*)buf_dyn_l1);
        }
#endif
        for (int32_t sk = 0; sk < 2; ++sk) {
          if (sk == 0) {
#ifdef QSA_K_AHEAD
            asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(1 + (k_tile & 1)));
#else
            asc_sync_intra_wait(PIPE_MTE1, (((task_block + 1) & 1) + 22));
            asc_sync_intra_wait(PIPE_MTE1, (((task_block + 1) & 1) + 6));
#endif
          }
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + QSA_K_OFFSET(task_block)), 0, (sk * 8), 8, 8, 8, 8);
          if (sk == 1) {
#ifdef QSA_K_AHEAD
            asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1 + (k_tile & 1)));
#else
            asc_sync_intra_arrive(PIPE_MTE1, (((task_block + 1) & 1) + 4));
            asc_sync_intra_arrive(PIPE_MTE1, (((task_block + 1) & 1) + 20));
#endif
          }
          if (sk == 0) {
            asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
          }
          asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 98304), 0, (sk * 8), 1, 8, 1, 1);
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          if (sk == 1) {
            asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
          }
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          asc_mmad(((__cc__ float*)buf_dyn_l0c + ((((task_block + 1) & 1) * 2048) + 8192)), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b), 16, 128, 128, 0, 1, 0, (sk == 0));
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          if (sk == 1) {
            asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(((task_block + 1) & 1)));
          }
        }
#if defined(QSA_K_AHEAD) && QSA_K_AHEAD
        // Current QK is queued. Stage the next K before the previous tile's PV.
        if (k_tile + 1 < tiles) {
          qsa_load_k_tile<CoreCount>(k_tile + 1, Indices, K, (__cbuf__ bfloat16_t*)buf_dyn_l1);
        }
#endif
      }
      if (34 <= task_block) {
        for (int32_t sk_1 = 0; sk_1 < 2; ++sk_1) {
          if (sk_1 == 0) {
            asc_sync_intra_wait(PIPE_MTE1, ((task_block & 1) + 18));
            asc_sync_intra_wait(PIPE_MTE1, ((task_block & 1) + 2));
          }
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
          asc_copy_l12l0b_transpose(((__cb__ bfloat16_t*)buf_dyn_l0b + 16384), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + (QSA_V_READ_SLOT(task_block) * 32768)), (sk_1 * 4), 0, 4, 16, 8, 16);
#ifdef QSA_K_AHEAD
          if (sk_1 == 1 && task_block - 34 + 2 < tiles) {
            asc_sync_intra_arrive(PIPE_MTE1, 4 + (task_block & 1));
            asc_sync_intra_arrive(PIPE_MTE1, 20 + (task_block & 1));
          }
#endif
          if (sk_1 == 0) {
            asc_sync_intra_wait(PIPE_MTE1, 25);
            asc_sync_intra_wait(PIPE_MTE1, 9);
          }
          asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a + 2048), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + (((task_block & 1) * 2048) + 102400)), 0, (sk_1 * 4), 1, 4, 1, 1);
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
          asc_mmad(((__cc__ float*)buf_dyn_l0c + ((task_block & 1) * 4096)), ((__ca__ bfloat16_t*)buf_dyn_l0a + 2048), ((__cb__ bfloat16_t*)buf_dyn_l0b + 16384), 16, 64, 256, 0, 1, 0, (sk_1 == 0));
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
          if (sk_1 == 1) {
            asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(2));
          }
        }
        asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(2));
        asc_set_l0c_copy_nz_para(1, 0, 0);
        asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[(((task_block & 1) * 2048) + 16640)])), ((__cc__ float*)buf_dyn_l0c + ((task_block & 1) * 4096)), 256, 16, 256, 16, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
        asc_sync_intra_arrive(PIPE_FIX, 8);
        asc_sync_intra_arrive(PIPE_FIX, 24);
      }
      if ((task_block < tiles + 33) && (33 <= task_block)) {
        asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(((task_block + 1) & 1)));
        asc_set_l0c_copy_nz_para(1, 0, 0);
        asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[((((task_block + 1) & 1) * 1024) + 20736)])), ((__cc__ float*)buf_dyn_l0c + ((((task_block + 1) & 1) * 2048) + 8192)), 128, 16, 128, 16, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
        asc_sync_intra_arrive(PIPE_FIX, ((task_block + 1) & 1));
        asc_sync_intra_arrive(PIPE_FIX, (((task_block + 1) & 1) + 16));
      }
    }
    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
#ifdef QSA_K_AHEAD
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
#endif
  }
  if ASC_IS_AIV {
    int32_t sid = asc_get_sub_block_id();
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(2));
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(3));
    // The first 111104 bytes are the old kernel's UB layout. The
    // additional rows are private to this AIV. Persistent blocks alternate
    // two rows so the next task's prefetch can overlap the previous V gather.
    auto row_indices = (__ubuf__ int32_t*)(buf_dyn_shmem + 111104);
    if constexpr (Prefetch && CoreCount == 0) {
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)row_indices,
          (__gm__ uint8_t*)(Indices + ((int32_t)block_idx / 2) * 2176),
          1, 8704, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 8704, 8704);
      asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));
      asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));
    } else {
      asc_sync_notify(PIPE_S, PIPE_MTE2, static_cast<event_t>(0));
      asc_sync_notify(PIPE_S, PIPE_MTE2, static_cast<event_t>(1));
    }
    KV_ub_version_counter_1 = 0;
    for (int32_t task_block_1 = 0; task_block_1 < tiles + 5; ++task_block_1) {
      if (5 <= task_block_1) {
        asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(2));
        if (CoreCount ? (task_block_1 - 5) % 17 == 16 : task_block_1 == 21) {
          asc_copy_ub2gm_align((__gm__ uint8_t*)((&(O[qsa_task_for_tile<CoreCount>(task_block_1 - 5) * 4096 + sid * 2048]))), (__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[51968]))), 1, 4096, static_cast<asc_store_l2_cache_mode>(0), 4096, 4096);
          if constexpr (CoreCount != 0) {
            asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(4));
          }
        }
      }
      if ((task_block_1 < tiles + 4) && (4 <= task_block_1)) {
        if (CoreCount ? (task_block_1 - 4) % 17 == 16 : task_block_1 == 20) {
          // Wait for the previous task's output DMA before reusing O_out.
          if (CoreCount != 0 && task_block_1 > 20) {
            asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(4));
          }
          main_kernel_simd_vf_3(buf_dyn_shmem);
        }
        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(2));
      }
      if ((task_block_1 < tiles + 3) && (3 <= task_block_1)) {
        if (CoreCount ? (task_block_1 - 3) % 17 == 0 : task_block_1 == 3) {
          main_kernel_simd_vf_0(buf_dyn_shmem);
        }
        asc_sync_intra_wait(PIPE_V, ((task_block_1 + 1) & 1));
#ifdef QSA_K_AHEAD
        // Removing the K relay also removes the indirect Valid -> QK -> V edge.
        asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(2 + ((task_block_1 + 1) & 1)));
#endif
        main_kernel_simd_vf_1(buf_dyn_shmem, task_block_1);
        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(3));
        asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((((task_block_1 + 1) & 1) + 2)));
        asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(3));
        asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)buf_dyn_l1 + (((((task_block_1 + 1) & 1) * 2048) + (sid * 128)) + 102400))), (__ubuf__ void*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((((task_block_1 + 1) & 1) * 1152) + 49664)]))), 8, 8, 1, 8);
        asc_sync_intra_arrive(PIPE_MTE3, 9);
      }
      if ((task_block_1 < tiles + 2) && (2 <= task_block_1)) {
#ifdef QSA_K_AHEAD
        // V now owns the first scalar read of the index tile.
        asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(task_block_1 & 1));
#endif
#ifdef QSA_PAIR_CHUNKS
        for (int issue = 0; issue < 4; ++issue) {
          int chunk = issue & 1;
#else
        for (int32_t chunk = 0; chunk < 2; ++chunk) {
#endif
          for (int32_t quad = 0; quad < QSA_GATHER_QUADS; ++quad) {
            int32_t base = Prefetch ? row_indices[qsa_row_slot_offset<CoreCount>(task_block_1 - 2) + sid * 64 + chunk * 32 + quad * 4]
                  : ((__ubuf__ int32_t*)buf_dyn_shmem)[((((((task_block_1 & 1) * 128) + (sid * 64)) + (chunk * 32)) + (quad * 4)) + 27008)];
            int8_t __cond_2 = ((int8_t)(bool)0);
            __cond_2 = ((int8_t)(base <= last_full_quad));
            bool skip_invalid = false;
#ifdef QSA_SKIP_TAIL
            // Valid for this tile was loaded into the matching parity slot
            // before this gather. Do not add another GM transaction here.
            skip_invalid = (((task_block_1 - 2) % 17) == 16);
#endif
            int32_t valid_quad = 1;
#ifdef QSA_SKIP_TAIL
            if (skip_invalid) {
              for (int32_t i = 0; i < 4; ++i) {
                valid_quad &= ((__ubuf__ int32_t*)buf_dyn_shmem)[((((((task_block_1 & 1) * 128) + (sid * 64)) + (chunk * 32)) + (quad * 4) + i) + 27264)];
              }
            }
#endif
            if (quad == 0) {
              asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(QSA_GATHER_SLOT(chunk)));
            }
            if ((bool)__cond_2 && (!skip_invalid || valid_quad != 0)) {
              asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((QSA_GATHER_SLOT(chunk) * 8192) + (quad * 1024)) + 16896)]))), (__gm__ uint8_t*)((&(V[((base * 512) + ((((int32_t)block_idx) & 1) * 256))]))), 4, 512, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 1024, 512);
            }
            if (!((bool)__cond_2) || (skip_invalid && valid_quad == 0)) {
              for (int32_t i = 0; i < 4; ++i) {
                int32_t idx = Prefetch ? row_indices[qsa_row_slot_offset<CoreCount>(task_block_1 - 2) + sid * 64 + chunk * 32 + quad * 4 + i]
                  : ((__ubuf__ int32_t*)buf_dyn_shmem)[(((((((task_block_1 & 1) * 128) + (sid * 64)) + (chunk * 32)) + (quad * 4)) + i) + 27008)];
                int32_t valid_row = 1;
#ifdef QSA_SKIP_TAIL
                if (skip_invalid) {
                  valid_row = ((__ubuf__ int32_t*)buf_dyn_shmem)[((((((task_block_1 & 1) * 128) + (sid * 64)) + (chunk * 32)) + (quad * 4) + i) + 27264)];
                }
                if (valid_row == 0) continue;
#endif
                asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((((QSA_GATHER_SLOT(chunk) * 8192) + (quad * 1024)) + (i * 256)) + 16896)]))), (__gm__ uint8_t*)((&(V[((idx * 512) + ((((int32_t)block_idx) & 1) * 256))]))), 1, 512, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 512, 512);
              }
            }
            if (quad == 7) {
              asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(QSA_GATHER_SLOT(chunk)));
            }
            if (!Prefetch && (chunk == 1) && (quad == 7)) {
              asc_sync_notify(PIPE_S, PIPE_MTE2, static_cast<event_t>((task_block_1 & 1)));
            }
            if (Prefetch && CoreCount && chunk == 1 && quad == 7 && (task_block_1 - 2) % 17 == 16) {
              asc_sync_notify(PIPE_S, PIPE_MTE2, static_cast<event_t>(((task_block_1 - 2) / 17) & 1));
            }
          }
#ifdef QSA_PAIR_CHUNKS
          if (issue < 2) continue;
#endif
          asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>((KV_ub_version_counter_1 & 1)));
          asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>((KV_ub_version_counter_1 & 1)));
          ascend_nd2nz_scatter<32, 256, 33, bfloat16_t, bfloat16_t>((__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((KV_ub_version_counter_1 & 1) * 8192) + 16896)]))), (__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((KV_ub_version_counter_1 & 1) * 8448)]))));
          asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>((KV_ub_version_counter_1 & 1)));
          asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((KV_ub_version_counter_1 & 1)));
#ifdef QSA_K_AHEAD
          if (chunk == 0 && task_block_1 >= 4) {  // the first two V slots have no predecessor
#else
          if (chunk == 0) {
#endif
            asc_sync_intra_wait(PIPE_MTE3, ((task_block_1 & 1) + 4));
          }
          asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>((KV_ub_version_counter_1 & 1)));
          asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)buf_dyn_l1 + (((QSA_V_WRITE_SLOT(task_block_1) * 32768) + (sid * 1024)) + (chunk * 512)))), (__ubuf__ void*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((KV_ub_version_counter_1 & 1) * 8448)]))), 16, 32, 1, 96);
          asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>((KV_ub_version_counter_1 & 1)));
          if (chunk == 1) {
            asc_sync_intra_arrive(PIPE_MTE3, ((task_block_1 & 1) + 2));
          }
          KV_ub_version_counter_1 = (KV_ub_version_counter_1 + 1);
        }
      }
      if ((task_block_1 < tiles + 1) && (1 <= task_block_1)) {
        asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>((((task_block_1 + 1) & 1) + 2)));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ int32_t*)buf_dyn_shmem)[((((task_block_1 + 1) & 1) * 128) + 27264)]))), (__gm__ uint8_t*)((&(Valid[CoreCount ? ((qsa_task_for_tile<CoreCount>(task_block_1 - 1) >> 1) * 2176 + ((task_block_1 - 1) % 17) * 128) : ((((((int32_t)block_idx) >> 1) * 2176) + (task_block_1 * 128)) - 128)]))), 1, 512, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 512, 512);
#ifdef QSA_K_AHEAD
        asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(2 + ((task_block_1 + 1) & 1)));
#else
#ifdef QSA_PAIR_CHUNKS
        for (int issue = 0; issue < 4; ++issue) {
          int chunk_1 = issue & 1;
#else
        for (int32_t chunk_1 = 0; chunk_1 < 2; ++chunk_1) {
#endif
          for (int32_t quad_1 = 0; quad_1 < QSA_GATHER_QUADS; ++quad_1) {
            if (!Prefetch && (chunk_1 == 0) && (quad_1 == 0)) {
              asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(((task_block_1 + 1) & 1)));
            }
            if (Prefetch && CoreCount && chunk_1 == 0 && quad_1 == 0 && (task_block_1 - 1) % 17 == 0) {
              asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(((task_block_1 - 1) / 17) & 1));
            }
            int32_t base_1 = Prefetch ? row_indices[qsa_row_slot_offset<CoreCount>(task_block_1 - 1) + sid * 64 + chunk_1 * 32 + quad_1 * 4]
                  : ((__ubuf__ int32_t*)buf_dyn_shmem)[(((((((task_block_1 + 1) & 1) * 128) + (sid * 64)) + (chunk_1 * 32)) + (quad_1 * 4)) + 27008)];
            int8_t __cond_1 = ((int8_t)(bool)0);
            __cond_1 = ((int8_t)(base_1 <= last_full_quad));
            bool skip_invalid_1 = false;
#ifdef QSA_SKIP_TAIL
            skip_invalid_1 = (((task_block_1 - 1) % 17) == 16);
#endif
            int32_t valid_quad_1 = 1;
#ifdef QSA_SKIP_TAIL
            if (skip_invalid_1) {
              for (int32_t i_1 = 0; i_1 < 4; ++i_1) {
                valid_quad_1 &= ((__ubuf__ int32_t*)buf_dyn_shmem)[(((((((task_block_1 + 1) & 1) * 128) + (sid * 64)) + (chunk_1 * 32)) + (quad_1 * 4) + i_1) + 27264)];
              }
            }
#endif
            if (quad_1 == 0) {
              asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(QSA_GATHER_SLOT(chunk_1)));
            }
            if ((bool)__cond_1 && (!skip_invalid_1 || valid_quad_1 != 0)) {
              asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((QSA_GATHER_SLOT(chunk_1) * 8192) + (quad_1 * 1024)) + 16896)]))), (__gm__ uint8_t*)((&(K[((base_1 * 512) + ((((int32_t)block_idx) & 1) * 256))]))), 4, 512, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 1024, 512);
            }
            if (!((bool)__cond_1) || (skip_invalid_1 && valid_quad_1 == 0)) {
              for (int32_t i_1 = 0; i_1 < 4; ++i_1) {
                int32_t idx_1 = Prefetch ? row_indices[qsa_row_slot_offset<CoreCount>(task_block_1 - 1) + sid * 64 + chunk_1 * 32 + quad_1 * 4 + i_1]
                  : ((__ubuf__ int32_t*)buf_dyn_shmem)[((((((((task_block_1 + 1) & 1) * 128) + (sid * 64)) + (chunk_1 * 32)) + (quad_1 * 4)) + i_1) + 27008)];
                int32_t valid_row_1 = 1;
#ifdef QSA_SKIP_TAIL
                if (skip_invalid_1) {
                  valid_row_1 = ((__ubuf__ int32_t*)buf_dyn_shmem)[(((((((task_block_1 + 1) & 1) * 128) + (sid * 64)) + (chunk_1 * 32)) + (quad_1 * 4) + i_1) + 27264)];
                }
                if (valid_row_1 == 0) continue;
#endif
                asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((((QSA_GATHER_SLOT(chunk_1) * 8192) + (quad_1 * 1024)) + (i_1 * 256)) + 16896)]))), (__gm__ uint8_t*)((&(K[((idx_1 * 512) + ((((int32_t)block_idx) & 1) * 256))]))), 1, 512, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 512, 512);
              }
            }
            if (quad_1 == 7) {
              asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(QSA_GATHER_SLOT(chunk_1)));
            }
          }
#ifdef QSA_PAIR_CHUNKS
          if (issue < 2) continue;
#endif
          asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>((KV_ub_version_counter_1 & 1)));
          asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>((KV_ub_version_counter_1 & 1)));
          ascend_nd2nz_scatter<32, 256, 33, bfloat16_t, bfloat16_t>((__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((KV_ub_version_counter_1 & 1) * 8192) + 16896)]))), (__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((KV_ub_version_counter_1 & 1) * 8448)]))));
          asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>((KV_ub_version_counter_1 & 1)));
          asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((KV_ub_version_counter_1 & 1)));
          asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>((KV_ub_version_counter_1 & 1)));
          asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)buf_dyn_l1 + (((((task_block_1 + 2) % 3) * 32768) + (sid * 1024)) + (chunk_1 * 512)))), (__ubuf__ void*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((KV_ub_version_counter_1 & 1) * 8448)]))), 16, 32, 1, 96);
          asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>((KV_ub_version_counter_1 & 1)));
          if (chunk_1 == 1) {
            asc_sync_intra_arrive(PIPE_MTE3, (((task_block_1 + 1) & 1) + 6));
          }
          KV_ub_version_counter_1 = (KV_ub_version_counter_1 + 1);
        }
#endif
      }
      if ((task_block_1 < tiles + 3) && (3 <= task_block_1)) {
        asc_sync_intra_wait(PIPE_V, 8);
        main_kernel_simd_vf_2(buf_dyn_shmem, task_block_1);
      }
      if (Prefetch && CoreCount && task_block_1 < tiles && task_block_1 % 17 == 0) {
        int slot = (task_block_1 / 17) & 1;
        int query = qsa_task_for_tile<CoreCount>(task_block_1) >> 1;
        asc_sync_wait(PIPE_S, PIPE_MTE2, static_cast<event_t>(slot));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)(row_indices + slot * 2176),
            (__gm__ uint8_t*)(Indices + query * 2176), 1, 8704, 0, 0, 0,
            static_cast<asc_load_l2_cache_mode>(0), 8704, 8704);
        asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(slot));
      }
      if (!Prefetch && task_block_1 < tiles) {
        asc_sync_wait(PIPE_S, PIPE_MTE2, static_cast<event_t>((task_block_1 & 1)));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ int32_t*)buf_dyn_shmem)[(((task_block_1 & 1) * 128) + 27008)]))), (__gm__ uint8_t*)((&(Indices[CoreCount ? ((qsa_task_for_tile<CoreCount>(task_block_1) >> 1) * 2176 + ((task_block_1) % 17) * 128) : (((((int32_t)block_idx) >> 1) * 2176) + (task_block_1 * 128))]))), 1, 512, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 512, 512);
        asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>((task_block_1 & 1)));
      }
    }
    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
    asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
    asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
    asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(2));
    asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(3));
    if constexpr (CoreCount != 0) {
      asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(4));
    }
    if constexpr (!Prefetch || CoreCount != 0) {
      asc_sync_wait(PIPE_S, PIPE_MTE2, static_cast<event_t>(0));
      asc_sync_wait(PIPE_S, PIPE_MTE2, static_cast<event_t>(1));
    }
  }
}

#undef QSA_GATHER_SLOT
#undef QSA_GATHER_QUADS
#undef QSA_SKIP_TAIL
#undef QSA_K_OFFSET
#undef QSA_V_READ_SLOT
#undef QSA_V_WRITE_SLOT

// MixedKernel emits unused sub-block-ID reads around the two expert calls.
// The ID is invariant on a subcore. Allow Bisheng to elide unused reads in
// this translation unit while preserving the real ID for any live use.
__aicore__ __attribute__((const)) static inline int qsa_sub_block_id() {
  return asc_get_sub_block_id();
}
#define asc_get_sub_block_id qsa_sub_block_id
