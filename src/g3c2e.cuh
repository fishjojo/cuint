#pragma once

#include <cstdint>
#include <cstdio>

#include "boys.cuh"
#include "cart2sph.h"
#include "config.h"
#include "cuint.h"
#include "hermite.cuh"
#include "macro.cuh"
#include "utils.h"

namespace g3c {

/*
template <int li, int lj, int lk>
__forceinline__ __device__ void write_integral(
    double *result, const double (&r)[md::nherm(li + lj + lk)],
    const double (&e_x)[md::nepair<li, lj>()],
    const double (&e_y)[md::nepair<li, lj>()],
    const double (&e_z)[md::nepair<li, lj>()],
    const double (&aux_power)[lk + 1], const int n_functions,
    const int n_aux) {
  constexpr auto Ci = c2s_matrix<li>();
  constexpr auto Bi = cart_ls<li>();
  constexpr auto Cj = c2s_matrix<lj>();
  constexpr auto Bj = cart_ls<lj>();
  constexpr auto Ck = c2s_matrix<lk>();
  constexpr auto Bk = cart_ls<lk>();

  static_for<0, 2 * li + 1>([&]<int ioff>() {
    static_for<0, 2 * lj + 1>([&]<int joff>() {
      double *ptr_out = result + (std::int64_t)(ioff * n_functions + joff) * n_aux;
      static_for<0, 2 * lk + 1>([&]<int koff>() {
        double value = 0;

        static_for<0, ncart(li)>([&]<int ic>() {
          constexpr double ci = Ci[ioff][ic];
          constexpr double abs_ci = (ci < 0.0) ? -ci : ci;
          if constexpr (abs_ci > 1e-15) {
            static_for<0, ncart(lj)>([&]<int jc>() {
              constexpr double cj = Cj[joff][jc];
              constexpr double abs_cj = (cj < 0.0) ? -cj : cj;
              if constexpr (abs_cj > 1e-15) {
                constexpr int ax = Bi[ic][0], ay = Bi[ic][1], az = Bi[ic][2];
                constexpr int bx = Bj[jc][0], by = Bj[jc][1], bz = Bj[jc][2];

                static_for<0, ax + bx + 1>([&]<int px>() {
                  static_for<0, ay + by + 1>([&]<int py>() {
                    static_for<0, az + bz + 1>([&]<int pz>() {

                      static_for<0, ncart(lk)>([&]<int kc>() {
                        constexpr double ck = Ck[koff][kc];
                        constexpr double abs_ck = (ck < 0.0) ? -ck : ck;
                        if constexpr (abs_ck > 1e-15) {
                          constexpr int cx = Bk[kc][0], cy = Bk[kc][1], cz = Bk[kc][2];

                          static_for<0, cx / 2 + 1>([&]<int tx>() {
                            constexpr int qx = cx - 2 * tx;
                            static_for<0, cy / 2 + 1>([&]<int ty>() {
                              constexpr int qy = cy - 2 * ty;
                              static_for<0, cz / 2 + 1>([&]<int tz>() {
                                constexpr int qz = cz - 2 * tz;
                                constexpr double coefficient =
                                    ck * md::aux_coef(cx, qx) * md::aux_coef(cy, qy) *
                                    md::aux_coef(cz, qz);
                                constexpr int power =
                                    (cx + qx) / 2 + (cy + qy) / 2 + (cz + qz) / 2;
                                value += ci * cj * coefficient * aux_power[power] *
                                         e_x[md::eindex<li, lj>(ax, bx, px)] *
                                         e_y[md::eindex<li, lj>(ay, by, py)] *
                                         e_z[md::eindex<li, lj>(az, bz, pz)] *
                                         r[md::hindex(px + qx, pz + qy, pz + qz)];
                              });
                            });
                          });
                        }
                      });
                    });
                  });
                });
              }
            });
          }
        });
        atomicAdd(ptr_out + koff, value);
      });
    });
  });
}
*/

// Fold the geometry-free aux side and the pair E coefficients into the
// Hermite tower and write the solid-harmonic (ij|k) tile. result points at
// the (i_function, j_function, k_function) element; aux is the fastest axis.
template <int li, int lj, int lk>
__forceinline__ __device__ void write_integral(
    double *result, const double (&r)[md::nherm(li + lj + lk)],
    const double (&e_x)[md::nepair<li, lj>()],
    const double (&e_y)[md::nepair<li, lj>()],
    const double (&e_z)[md::nepair<li, lj>()],
    const double (&aux_power)[lk + 1], const int n_functions,
    const int n_aux) {
  constexpr auto Ci = c2s_matrix<li>();
  constexpr auto Bi = cart_ls<li>();
  constexpr auto Cj = c2s_matrix<lj>();
  constexpr auto Bj = cart_ls<lj>();
  constexpr auto Ck = c2s_matrix<lk>();
  constexpr auto Bk = cart_ls<lk>();

  double s[md::nherm(li + lj)];
  double tile[ncart(li) * ncart(lj)];

  static_for<0, 2 * lk + 1>([&]<int koff>() {
    // ket Hermite to spherical
    static_for<0, md::nherm(li + lj)>([&]<int ip>() {
      constexpr md::Triple p = md::herm_triple(ip);
      double value = 0.0;

      static_for<0, ncart(lk)>([&]<int kc>() {
        constexpr double ck = Ck[koff][kc];
        constexpr double abs_ck = (ck < 0.0) ? -ck : ck;
        if constexpr (abs_ck > 1e-15) {
          constexpr int cx = Bk[kc][0], cy = Bk[kc][1], cz = Bk[kc][2];
          static_for<0, cx / 2 + 1>([&]<int tx>() {
            constexpr int qx = cx - 2 * tx;
            static_for<0, cy / 2 + 1>([&]<int ty>() {
              constexpr int qy = cy - 2 * ty;
              static_for<0, cz / 2 + 1>([&]<int tz>() {
                constexpr int qz = cz - 2 * tz;
                constexpr double coefficient =
                    ck * md::aux_coef(cx, qx) * md::aux_coef(cy, qy) *
                    md::aux_coef(cz, qz);
                constexpr int power =
                    (cx + qx) / 2 + (cy + qy) / 2 + (cz + qz) / 2;
                value += coefficient * aux_power[power] *
                         r[md::hindex(p.t + qx, p.u + qy, p.v + qz)];
              });
            });
          });
        }
      });

      s[ip] = value;
    });

    // bra Hermite to Cartesian
    static_for<0, ncart(li)>([&]<int ic>() {
      constexpr int ax = Bi[ic][0], ay = Bi[ic][1], az = Bi[ic][2];
      static_for<0, ncart(lj)>([&]<int jc>() {
        constexpr int bx = Bj[jc][0], by = Bj[jc][1], bz = Bj[jc][2];
        double value = 0.0;

        static_for<0, ax + bx + 1>([&]<int px>() {
          static_for<0, ay + by + 1>([&]<int py>() {
            static_for<0, az + bz + 1>([&]<int pz>() {
              value += e_x[md::eindex<li, lj>(ax, bx, px)] *
                       e_y[md::eindex<li, lj>(ay, by, py)] *
                       e_z[md::eindex<li, lj>(az, bz, pz)] *
                       s[md::hindex(px, py, pz)];
            });
          });
        });

        tile[ic * ncart(lj) + jc] = value;
      });
    });

    // bra cart2sph and write
    static_for<0, 2 * li + 1>([&]<int ioff>() {
      static_for<0, 2 * lj + 1>([&]<int joff>() {
        double value = 0.0;

        static_for<0, ncart(li)>([&]<int ic>() {
          constexpr double ci = Ci[ioff][ic];
          constexpr double abs_ci = (ci < 0.0) ? -ci : ci;
          if constexpr (abs_ci > 1e-15) {
            static_for<0, ncart(lj)>([&]<int jc>() {
              constexpr double cj = Cj[joff][jc];
              constexpr double abs_cj = (cj < 0.0) ? -cj : cj;
              if constexpr (abs_cj > 1e-15) {
                constexpr double cij = ci * cj;
                value += cij * tile[ic * ncart(lj) + jc];
              }
            });
          }
        });

        atomicAdd(result + (std::int64_t)(ioff * n_functions + joff) * n_aux + koff,
                  value);
      });
    });
  });
}

template <int li, int lj, int lk>
__global__ void kernel(double *result, const int *pair_indices,
                       const int n_pairs, const int n_primitives,
                       const int *primitive_to_function, const int n_functions,
                       const int *aux_indices,
                       const int *aux_primitive_to_function, const int n_aux,
                       const int *atm, const int atm_stride, const int *bas,
                       const int bas_stride, const double *env,
                       const int env_stride, const int is_screened) {
  atm += blockIdx.z * atm_stride;
  bas += blockIdx.z * bas_stride;
  env += blockIdx.z * env_stride;
  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_idx >= n_pairs) return;
  int i_primitive, j_primitive;

  if (is_screened) {
    const int primitive_pair = pair_indices[pair_idx];
    i_primitive = primitive_pair / n_primitives;
    j_primitive = primitive_pair % n_primitives;
  } else {
    const int i_begin = pair_indices[0];
    const int i_end = pair_indices[1];
    const int j_begin = pair_indices[2];
    const int j_end = pair_indices[3];
    const int n_rows = i_end - i_begin;
    const int n_cols = j_end - j_begin;
    if constexpr (li == lj) {
      const double sqrt_target =
          (2 * n_cols + 1) * (2 * n_cols + 1) - 8 * pair_idx;
      i_primitive = (int)floor((2 * n_cols - 1 - sqrt(sqrt_target)) / 2) + 1;
      j_primitive =
          pair_idx - (2 * n_cols - i_primitive - 1) * i_primitive / 2;
    } else {
      const int stride = max(n_rows, n_cols);
      const int index_with_larger_stride = pair_idx / stride;
      const int index_with_smaller_stride = pair_idx % stride;
      i_primitive = n_rows >= n_cols ? index_with_smaller_stride
                                     : index_with_larger_stride;
      j_primitive = n_rows >= n_cols ? index_with_larger_stride
                                     : index_with_smaller_stride;
    }
    i_primitive += i_begin;
    j_primitive += j_begin;
  }

  const double alpha = env[bas(PTR_EXP, i_primitive)];
  const double beta = env[bas(PTR_EXP, j_primitive)];
  const double c1 = env[bas(PTR_COEFF, i_primitive)];
  const double c2 = env[bas(PTR_COEFF, j_primitive)];
  const int i_atom = bas(ATOM_OF, i_primitive);
  const int j_atom = bas(ATOM_OF, j_primitive);
  const int i_coord_offset = atm(PTR_COORD, i_atom);
  const int j_coord_offset = atm(PTR_COORD, j_atom);
  const double i_x = env[i_coord_offset + 0];
  const double i_y = env[i_coord_offset + 1];
  const double i_z = env[i_coord_offset + 2];
  const double j_x = env[j_coord_offset + 0];
  const double j_y = env[j_coord_offset + 1];
  const double j_z = env[j_coord_offset + 2];
  const double ix_to_jx = j_x - i_x;
  const double iy_to_jy = j_y - i_y;
  const double iz_to_jz = j_z - i_z;
  const double pair_distance_squared =
      ix_to_jx * ix_to_jx + iy_to_jy * iy_to_jy + iz_to_jz * iz_to_jz;
  const double pair_exponent = alpha + beta;

  double prefactor = c1 * c2;
  if (i_primitive == j_primitive) {
    prefactor *= 0.5;
  }

  // aux primitive
  const int k_primitive = aux_indices[blockIdx.y];
  const double gamma = env[bas(PTR_EXP, k_primitive)];
  prefactor *= env[bas(PTR_COEFF, k_primitive)]; // c3
  const int k_atom = bas(ATOM_OF, k_primitive);
  const int k_coord_offset = atm(PTR_COORD, k_atom);
  const double k_x = env[k_coord_offset + 0];
  const double k_y = env[k_coord_offset + 1];
  const double k_z = env[k_coord_offset + 2];

  // Gaussian product center of the pair and its distance to the aux center
  const double a_weight = alpha / pair_exponent;
  const double b_weight = beta / pair_exponent;
  const double p_x = a_weight * i_x + b_weight * j_x;
  const double p_y = a_weight * i_y + b_weight * j_y;
  const double p_z = a_weight * i_z + b_weight * j_z;
  const double w_x = p_x - k_x;
  const double w_y = p_y - k_y;
  const double w_z = p_z - k_z;

  const double rho = pair_exponent * gamma / (pair_exponent + gamma);

  constexpr int L = li + lj + lk;
  double f[L + 1];
  boys_function<L>(rho * (w_x * w_x + w_y * w_y + w_z * w_z), f);

  // [0]^(m) = prefactor * (-2 rho)^m F_m with
  // prefactor = c1 c2 c3 exp(-mu |AB|^2) 2 pi^{5/2} / (zeta_p zeta_c
  // sqrt(zeta_p + zeta_c)); the ket Hermite parity collapses to (-1)^lk
  const double mu = alpha * beta / pair_exponent;
  prefactor *= exp(-mu * pair_distance_squared) * 34.98683665524972569 /
               (pair_exponent * gamma * sqrt(pair_exponent + gamma));
  if constexpr (lk % 2 == 1) {
    prefactor = -prefactor;
  }

  double scale = prefactor;
  static_for<0, L + 1>([&]<int m>() {
    f[m] *= scale;
    scale *= -2.0 * rho;
  });

  double r[md::nherm(L)];
  md::hermite_tower<L>(r, f, w_x, w_y, w_z);

  // pair expansion coefficients per axis, seeded with 1
  const double half_over_zeta = 0.5 / pair_exponent;
  double e_x[md::nepair<li, lj>()];
  double e_y[md::nepair<li, lj>()];
  double e_z[md::nepair<li, lj>()];
  // P - A = w_b (B - A), P - B = -w_a (B - A)
  md::e_pair<li, lj>(e_x, b_weight * ix_to_jx, -a_weight * ix_to_jx,
                     half_over_zeta);
  md::e_pair<li, lj>(e_y, b_weight * iy_to_jy, -a_weight * iy_to_jy,
                     half_over_zeta);
  md::e_pair<li, lj>(e_z, b_weight * iz_to_jz, -a_weight * iz_to_jz,
                     half_over_zeta);

  // powers of 1 / (2 zeta_c) for the single-center coefficients
  double aux_power[lk + 1];
  aux_power[0] = 1.0;
  const double half_over_gamma = 0.5 / gamma;
  static_for<0, lk>([&]<int i>() {
    aux_power[i + 1] = aux_power[i] * half_over_gamma;
  });

  const int i_function_index = primitive_to_function[i_primitive];
  const int j_function_index = primitive_to_function[j_primitive];
  const int k_function_index = aux_primitive_to_function[blockIdx.y];

  result += (std::int64_t)blockIdx.z * n_functions * n_functions * n_aux +
            (std::int64_t)(i_function_index * n_functions + j_function_index) * n_aux +
            k_function_index;

  write_integral<li, lj, lk>(result, r, e_x, e_y, e_z, aux_power, n_functions,
                             n_aux);
}

// Launch every (li <= lj) pair class for one aux angular momentum, bounded
// by the total angular momentum cap (classes above it belong to the
// two-stage path). Explicitly instantiated once per lk in g3c2e_ints.cu so
// the heavy kernel unrolling is spread over parallel translation units.
template <int lk>
void launch(cudaStream_t stream, double *result, const int *pair_indices,
            const int n_pairs, const int n_primitives,
            const int *primitive_to_function, const int n_functions,
            const int *aux_indices, const int n_aux_primitives,
            const int *aux_primitive_to_function, const int n_aux,
            const int *atm, const int atm_stride, const int *bas,
            const int bas_stride, const double *env, const int env_stride,
            const int n_configurations, const int i_angular,
            const int j_angular, const int is_screened) {
  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_aux_primitives,
                        (uint)n_configurations};

  dispatch_range<0, G3cPairLMAX1>(i_angular, [&]<int li>() {
    dispatch_range<li, G3cPairLMAX1>(j_angular, [&]<int lj>() {
      if constexpr (li + lj + lk <= G3cTotalLMAX) {
        kernel<li, lj, lk><<<block_grid, block_size, 0, stream>>>(
            result, pair_indices, n_pairs, n_primitives,
            primitive_to_function, n_functions, aux_indices,
            aux_primitive_to_function, n_aux, atm, atm_stride, bas,
            bas_stride, env, env_stride, is_screened);
      } else {
        std::fprintf(stderr,
                     "int3c2e_fused: class (%d,%d|%d) exceeds the compiled "
                     "total angular momentum cap %d\n",
                     li, lj, lk, G3cTotalLMAX);
      }
    });
  });
}

// signature of launch<lk>, shared by the extern template declarations and the
// explicit instantiations
#define G3C2E_LAUNCH_PARAMS                                                 \
  (cudaStream_t, double *, const int *, const int, const int, const int *,  \
   const int, const int *, const int, const int *, const int, const int *,  \
   const int, const int *, const int, const double *, const int, const int, \
   const int, const int, const int)

}  // namespace g3c
