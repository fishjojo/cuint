#include <math.h>

#include "config.h"
#include "cuint.h"
#include "macro.cuh"
#include "recursion.cuh"
#include "utils.h"
#include "write.cuh"

namespace ovlp {
template <int i_angular, int j_angular>
__global__ void kernel(double *result, const int *pair_indices,
                       const int n_primitives, const int n_pairs,
                       const int *primitive_to_function, const int n_functions,
                       const int *atm, const int atm_stride, const int *bas,
                       const int bas_stride, const double *env,
                       const int env_stride, const int is_screened) {
  OVLP_SPELL;

  result += blockIdx.y * n_functions * n_functions +
            i_function_index * n_functions + j_function_index;

  double x_pairs[(i_angular + 1) * (j_angular + 1)];
  reset(x, 0, 0);

  double y_pairs[(i_angular + 1) * (j_angular + 1)];
  reset(y, 0, 0);

  double z_pairs[(i_angular + 1) * (j_angular + 1)];
  reset(z, 0, 0);

  write(0);
}

template <int i_angular, int j_angular, int i_deriv, int j_deriv>
__global__ void gen_kernel(double *result, const int *pair_indices,
                           const int n_primitives, const int n_pairs,
                           const int *primitive_to_function,
                           const int n_functions, const int *atm,
                           const int atm_stride, const int *bas,
                           const int bas_stride, const double *env,
                           const int env_stride, const int is_screened,
                           const int comp) {
  OVLP_SPELL;

  result += blockIdx.y * comp * n_functions * n_functions +
            i_function_index * n_functions + j_function_index;

  double x_pairs[(i_angular + 1 + i_deriv) * (j_angular + 1 + j_deriv)];
  reset(x, i_deriv, j_deriv);

  double y_pairs[(i_angular + 1 + i_deriv) * (j_angular + 1 + j_deriv)];
  reset(y, i_deriv, j_deriv);

  double z_pairs[(i_angular + 1 + i_deriv) * (j_angular + 1 + j_deriv)];
  reset(z, i_deriv, j_deriv);

  constexpr int stride = j_angular + 1 + j_deriv;
  constexpr int n_slots = i_deriv + j_deriv;

  // Output components enumerate the Cartesian product of i_deriv copies of
  // (xi, yi, zi) followed by j_deriv copies of (xj, yj, zj), the leftmost
  // slot varying slowest: e.g. for i_deriv = 2, j_deriv = 0 the order is
  // xixi, xiyi, xizi, yixi, ..., zizi; for i_deriv = 1, j_deriv = 1 it is
  // xixj, xiyj, xizj, yixj, ..., zizj.
  static_for<0, pow3(n_slots)>([&]<int c>() {
    constexpr int ix = deriv_count(c, n_slots, 0, i_deriv, 0);
    constexpr int iy = deriv_count(c, n_slots, 0, i_deriv, 1);
    constexpr int iz = deriv_count(c, n_slots, 0, i_deriv, 2);
    constexpr int jx = deriv_count(c, n_slots, i_deriv, n_slots, 0);
    constexpr int jy = deriv_count(c, n_slots, i_deriv, n_slots, 1);
    constexpr int jz = deriv_count(c, n_slots, i_deriv, n_slots, 2);

    // restore the axes dirtied by the previous component
    if constexpr (c > 0) {
      if constexpr (deriv_count(c - 1, n_slots, 0, n_slots, 0) > 0) {
        reset(x, i_deriv, j_deriv);
      }
      if constexpr (deriv_count(c - 1, n_slots, 0, n_slots, 1) > 0) {
        reset(y, i_deriv, j_deriv);
      }
      if constexpr (deriv_count(c - 1, n_slots, 0, n_slots, 2) > 0) {
        reset(z, i_deriv, j_deriv);
      }
    }

    // Each application adds one derivative order. Application t (of ix on
    // this axis) only needs rows up to i_angular + ix - 1 - t valid
    // afterwards, so later applications skip the top rows; bra applications
    // keep the extra jx columns alive for the ket applications that follow.
    static_for<0, ix>([&]<int t>() {
      rr::nabla1i_1e<i_angular + ix - 1 - t, j_angular + jx, stride>(x_pairs,
                                                                     2 * alpha);
    });
    static_for<0, iy>([&]<int t>() {
      rr::nabla1i_1e<i_angular + iy - 1 - t, j_angular + jy, stride>(y_pairs,
                                                                     2 * alpha);
    });
    static_for<0, iz>([&]<int t>() {
      rr::nabla1i_1e<i_angular + iz - 1 - t, j_angular + jz, stride>(z_pairs,
                                                                     2 * alpha);
    });

    static_for<0, jx>([&]<int t>() {
      rr::nabla1j_1e<i_angular, j_angular + jx - 1 - t, stride>(x_pairs,
                                                                2 * beta);
    });
    static_for<0, jy>([&]<int t>() {
      rr::nabla1j_1e<i_angular, j_angular + jy - 1 - t, stride>(y_pairs,
                                                                2 * beta);
    });
    static_for<0, jz>([&]<int t>() {
      rr::nabla1j_1e<i_angular, j_angular + jz - 1 - t, stride>(z_pairs,
                                                                2 * beta);
    });

    write(j_deriv);
  });
}

template <int i_angular, int j_angular>
__global__ void gradient(double *result, const int *pair_indices,
                         const int n_primitives, const int n_pairs,
                         const int *primitive_to_function,
                         const int n_functions, const int *atm,
                         const int atm_stride, const int *bas,
                         const int bas_stride, const double *env,
                         const int env_stride, const int is_screened) {
  OVLP_SPELL;

  result += blockIdx.y * 3 * n_functions * n_functions +
            i_function_index * n_functions + j_function_index;

  double x_pairs[(i_angular + 1) * (j_angular + 2)];
  reset(x, 0, 1);

  double y_pairs[(i_angular + 1) * (j_angular + 2)];
  reset(y, 0, 1);

  double z_pairs[(i_angular + 1) * (j_angular + 2)];
  reset(z, 0, 1);

  // x component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(x_pairs,
                                                                    2 * beta);
  write(1);
  reset(x, 0, 1);

  // y component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(y_pairs,
                                                                    2 * beta);
  write(1);
  reset(y, 0, 1);

  // z component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(z_pairs,
                                                                    2 * beta);
  write(1);
}
} // namespace ovlp

void overlap(cudaStream_t stream, double *result, const int *pair_indices,
             const int n_pairs, const int n_primitives,
             const int *primitive_to_function, const int n_functions,
             const int *atm, const int atm_stride, const int *bas,
             const int bas_stride, const double *env, const int env_stride,
             const int n_configurations, const int i_angular,
             const int j_angular, const int is_screened) {
  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_configurations,
                        1};

  dispatch_range<0, LMAX1>(i_angular, [&]<int li>() {
    dispatch_range<li, LMAX1>(j_angular, [&]<int lj>() {
      ovlp::kernel<li, lj><<<block_grid, block_size, 0, stream>>>(
          result, pair_indices, n_primitives, n_pairs, primitive_to_function,
          n_functions, atm, atm_stride, bas, bas_stride, env, env_stride,
          is_screened);
    });
  });
}

void overlap_gradient(cudaStream_t stream, double *result,
                      const int *pair_indices, const int n_pairs,
                      const int n_primitives, const int *primitive_to_function,
                      const int n_functions, const int *atm,
                      const int atm_stride, const int *bas,
                      const int bas_stride, const double *env,
                      const int env_stride, const int n_configurations,
                      const int i_angular, const int j_angular,
                      const int is_screened) {
  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_configurations,
                        1};

  dispatch_range<0, LMAX1>(i_angular, [&]<int li>() {
    dispatch_range<li, LMAX1>(j_angular, [&]<int lj>() {
      ovlp::gradient<li, lj><<<block_grid, block_size, 0, stream>>>(
          result, pair_indices, n_primitives, n_pairs, primitive_to_function,
          n_functions, atm, atm_stride, bas, bas_stride, env, env_stride,
          is_screened);
    });
  });
}

void gen_overlap(cudaStream_t stream, double *result, const int *pair_indices,
                 const int n_pairs, const int n_primitives,
                 const int *primitive_to_function, const int n_functions,
                 const int *atm, const int atm_stride, const int *bas,
                 const int bas_stride, const double *env, const int env_stride,
                 const int n_configurations, const int i_angular,
                 const int j_angular, const int is_screened, const int i_deriv,
                 const int j_deriv, const int comp) {
  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_configurations,
                        1};

  dispatch_range<0, LMAX1>(i_angular, [&]<int li>() {
    dispatch_range<0, LMAX1>(j_angular, [&]<int lj>() {
      dispatch_range<0, DerivMAX1>(i_deriv, [&]<int di>() {
        // MAX_DERIV bounds the total derivative order i_deriv + j_deriv
        dispatch_range<0, DerivMAX1 - di>(j_deriv, [&]<int dj>() {
          ovlp::gen_kernel<li, lj, di, dj>
              <<<block_grid, block_size, 0, stream>>>(
                  result, pair_indices, n_primitives, n_pairs,
                  primitive_to_function, n_functions, atm, atm_stride, bas,
                  bas_stride, env, env_stride, is_screened, comp);
        });
      });
    });
  });
}
