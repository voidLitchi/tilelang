// Included only for the explicit hybrid schedules; units here are BF16 elements.
#include <tl_templates/ascend/common.h>

constexpr int qsa_k_tile_size = 128 * 256;
constexpr int qsa_k_l1_base = 102400 + 2 * 16 * 128;  // after both P slots

template <int CoreCount>
__aicore__ inline void qsa_load_k_tile(int tile, __gm__ int32_t* indices, __gm__ bfloat16_t* key,
                                      __cbuf__ bfloat16_t* l1) {
  int task = (int)block_idx + (tile / 17) * CoreCount;
  int slot = tile & 1;
  auto event = static_cast<event_t>(1 + slot);
  auto bases = indices + (task >> 1) * 2176 + (tile % 17) * 128;
  auto dst = l1 + qsa_k_l1_base + slot * qsa_k_tile_size;
  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, event);
  asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(128), 0);
  for (int quad = 0; quad < 32; ++quad) {
    // The adapter provides whole quads, a masked tail and three guard rows.
    auto src = key + bases[4 * quad] * 512 + (task & 1) * 256;
    asc_copy_gm2l1_nd2nz(dst + quad * 64, src, 1024, static_cast<asc_load_l2_cache_mode>(0), 4, 256, 0, 0);
  }
  asc_sync_notify(PIPE_MTE2, PIPE_MTE1, event);
}
