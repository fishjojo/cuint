// Explicit instantiation shard: compiled once per aux angular momentum with
// -DG3C2E_INST_LK=<lk> so the template-unrolled kernels build in parallel.
#include "g3c2e.cuh"

namespace g3c {

template void launch<G3C2E_INST_LK> G3C2E_LAUNCH_PARAMS;

}  // namespace g3c
