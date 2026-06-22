#include <math.h>
#include "cuint.h"
#include "pbc_macro.cuh"
#include "recursion.cuh"
#include "write.cuh"

namespace ovlp {
template <int i_angular, int j_angular>

__global__ void pbc_kernel(double *result, const int *pair_indices,
                           const int n_primitives, const int n_pairs,
                           const int *primitive_to_function, const int n_functions,
                           const int *atm, const int atm_stride, const int *bas,
                           const int bas_stride, const double *env, const int env_stride,
                           const double *Ls, const int *mask,
                           const int is_screened, const int reduce_over_images) {

  const int matrix_stride = n_functions * n_functions;

  OVLP_SPELL;

  if constexpr (i_angular == 0 && j_angular == 0) {
    atomicAdd(result, prefactor * prefactor * prefactor);
  } else {
    double x_pairs[(i_angular + 1) * (j_angular + 1)];
    reset(x, 0, 0);

    double y_pairs[(i_angular + 1) * (j_angular + 1)];
    reset(y, 0, 0);

    double z_pairs[(i_angular + 1) * (j_angular + 1)];
    reset(z, 0, 0);

    write(0);
  }
}

template <int i_angular, int j_angular>
__global__ void
pbc_gradient(double *result, const int *pair_indices, const int n_primitives,
             const int n_pairs, const int *primitive_to_function,
             const int n_functions, const int *atm, const int atm_stride,
             const int *bas, const int bas_stride, const double *env,
             const int env_stride, const double *Ls, const int *mask,
             const int is_screened, const int reduce_over_images) {

  const int matrix_stride = 3 * n_functions * n_functions;

  OVLP_SPELL;

  double x_pairs[(i_angular + 1) * (j_angular + 2)];
  reset(x, 0, 1);

  double y_pairs[(i_angular + 1) * (j_angular + 2)];
  reset(y, 0, 1);

  double z_pairs[(i_angular + 1) * (j_angular + 2)];
  reset(z, 0, 1);

  // // x component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(x_pairs,
                                                                    2 * beta);
  write(1);
  reset(x, 0, 1);

  // // y component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(y_pairs,
                                                                    2 * beta);
  write(1);
  reset(y, 0, 1);

  // // z component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(z_pairs,
                                                                    2 * beta);
  write(1);
}
} // namespace ovlp

void pbc_overlap(cudaStream_t stream,
                 double *result, const int *pair_indices, const int n_pairs,
                 const int n_primitives, const int *primitive_to_function,
                 const int n_functions, const int *atm, const int atm_stride,
                 const int *bas, const int bas_stride, const double *env,
                 const int env_stride, const int n_configurations,
                 const double *Ls, // (n_configurations, n_images, 3)
                 const int n_images,
                 const int *mask,  // (n_configurations, n_images)
                 const int i_angular, const int j_angular,
                 const int is_screened, const int reduce_over_images) {

  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_configurations,
                        (uint)n_images};

  switch (i_angular * 10 + j_angular) { tabulate_kernel(ovlp::pbc_kernel); }
}

void pbc_overlap_gradient(cudaStream_t stream,
                          double *result, const int *pair_indices,
                          const int n_pairs, const int n_primitives,
                          const int *primitive_to_function, const int n_functions,
                          const int *atm, const int atm_stride, const int *bas,
                          const int bas_stride, const double *env,
                          const int env_stride, const int n_configurations,
                          const double *Ls, // (n_configurations, n_images, 3)
                          const int n_images,
                          const int *mask,  // (n_configurations, n_images)
                          const int i_angular, const int j_angular,
                          const int is_screened, const int reduce_over_images) {

  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_configurations,
                        (uint)n_images};

  switch (i_angular * 10 + j_angular) { tabulate_kernel(ovlp::pbc_gradient); }
}
