// Unified 3c2e entry point: route one (i_angular, j_angular, k_angular)
// class to the fused single-kernel path or the two-stage path based on the
// angular momenta. Both paths take the same primitive-level metadata; the
// two-stage branch recovers the contracted shell structure on the host from
// the runs of equal first-function values (primitives of one contracted
// shell are consecutive and share their first function).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "config.h"
#include "cuint.h"

namespace {

// In-place symmetrization of the (config, i, j, aux) result: both (i, j)
// and (j, i) tiles become their sum, equivalent to
// result + result.transpose(0, 2, 1, 3) without the second allocation.
// Diagonal i == j entries double, matching the transpose-add because the
// integral kernels halve diagonal tiles. Blocks with j > i exit before
// touching memory.
__global__ void symmetrize_kernel(double *result, const int n_functions,
                                  const int n_aux,
                                  const int n_configurations) {
  const int j = blockIdx.y;
  const int i = blockIdx.z;
  if (j > i) return;
  const std::int64_t k =
      (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_aux) return;

  const std::int64_t config_stride =
      (std::int64_t)n_functions * n_functions * n_aux;
  double *lower =
      result + ((std::int64_t)i * n_functions + j) * n_aux + k;
  double *upper =
      result + ((std::int64_t)j * n_functions + i) * n_aux + k;
  for (int config = 0; config < n_configurations; config++) {
    const double sum =
        lower[config * config_stride] + upper[config * config_stride];
    lower[config * config_stride] = sum;
    upper[config * config_stride] = sum;
  }
}

// effective fused caps; classes above them route to the two-stage path
int fused_l_pair = G3cPairLMAX1 - 1;
int fused_l_aux = G3cAuxLMAX1 - 1;
int fused_l_total = G3cTotalLMAX;

}  // namespace

int int3c2e(cudaStream_t stream, double *result, const int *pair_indices,
            const int n_pairs, const int n_primitives,
            const int *primitive_to_function, const int n_functions,
            const int *aux_indices, const int n_aux_primitives,
            const int *aux_primitive_to_function, const int n_aux,
            const int *atm, const int atm_stride, const int *bas,
            const int bas_stride, const double *env, const int env_stride,
            const int n_configurations, const int i_angular,
            const int j_angular, const int k_angular, const int is_screened,
            const Int3c2eTsPlan *ts_plan, double *workspace,
            const size_t workspace_bytes) {
  if (i_angular <= fused_l_pair && j_angular <= fused_l_pair &&
      k_angular <= fused_l_aux &&
      i_angular + j_angular + k_angular <= fused_l_total) {
    int3c2e_fused(stream, result, pair_indices, n_pairs, n_primitives,
                  primitive_to_function, n_functions, aux_indices,
                  n_aux_primitives, aux_primitive_to_function, n_aux, atm,
                  atm_stride, bas, bas_stride, env, env_stride,
                  n_configurations, i_angular, j_angular, k_angular,
                  is_screened);
    return 0;
  }
  if (is_screened) {
    std::fprintf(stderr,
                 "int3c2e: screening is not supported above the fused caps "
                 "(class (%d,%d|%d))\n",
                 i_angular, j_angular, k_angular);
    return 7;
  }
  if (ts_plan) {
    return int3c2e_two_stage_planned(
        stream, result, ts_plan, aux_indices, n_aux_primitives, n_functions,
        n_aux, atm, atm_stride, bas, bas_stride, env, env_stride,
        n_configurations, i_angular, j_angular, k_angular, workspace);
  }
  std::fprintf(stderr,
               "int3c2e: class (%d,%d|%d) is above the fused caps and no "
               "ts_plan was given\n",
               i_angular, j_angular, k_angular);
  return 2;
}

void int3c2e_set_fused_caps(const int max_l_pair, const int max_l_aux,
                            const int max_l_total) {
  fused_l_pair = max_l_pair < G3cPairLMAX1 - 1 ? max_l_pair
                                               : G3cPairLMAX1 - 1;
  fused_l_aux = max_l_aux < G3cAuxLMAX1 - 1 ? max_l_aux : G3cAuxLMAX1 - 1;
  fused_l_total = max_l_total < G3cTotalLMAX ? max_l_total : G3cTotalLMAX;
}

void int3c2e_symmetrize(cudaStream_t stream, double *result,
                        const int n_configurations, const int n_functions,
                        const int n_aux) {
  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_aux + 255) / 256), (uint)n_functions,
                        (uint)n_functions};
  symmetrize_kernel<<<block_grid, block_size, 0, stream>>>(
      result, n_functions, n_aux, n_configurations);
}
