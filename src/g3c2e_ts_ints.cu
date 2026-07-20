// Explicit instantiation shard for the two-stage Phase B launchers,
// compiled once per aux angular momentum with -DG3C2E_TS_INST_LK=<lk> so
// the tower unrolling builds in parallel (mirrors g3c2e_ints.cu).
#include "g3c2e_ts.cuh"

namespace g3c::ts {

#define G3C2E_TS_INST_LAB(LAB) \
  template void launch_hermite_aux<LAB, G3C2E_TS_INST_LK> \
      G3C2E_TS_LAUNCH_B_PARAMS;

G3C2E_TS_INST_LAB(0)
#if 2 * CUINT_G3C2E_TS_MAX_L_PAIR >= 1
G3C2E_TS_INST_LAB(1)
#endif
#if 2 * CUINT_G3C2E_TS_MAX_L_PAIR >= 2
G3C2E_TS_INST_LAB(2)
#endif
#if 2 * CUINT_G3C2E_TS_MAX_L_PAIR >= 3
G3C2E_TS_INST_LAB(3)
#endif
#if 2 * CUINT_G3C2E_TS_MAX_L_PAIR >= 4
G3C2E_TS_INST_LAB(4)
#endif
#if 2 * CUINT_G3C2E_TS_MAX_L_PAIR >= 5
G3C2E_TS_INST_LAB(5)
#endif
#if 2 * CUINT_G3C2E_TS_MAX_L_PAIR >= 6
G3C2E_TS_INST_LAB(6)
#endif
#if 2 * CUINT_G3C2E_TS_MAX_L_PAIR >= 7
G3C2E_TS_INST_LAB(7)
#endif
#if 2 * CUINT_G3C2E_TS_MAX_L_PAIR >= 8
G3C2E_TS_INST_LAB(8)
#endif
#if 2 * CUINT_G3C2E_TS_MAX_L_PAIR >= 9
#error "add Phase B instantiations for G3C2E_TS_MAX_L_PAIR > 4"
#endif

}  // namespace g3c::ts
