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

// Two-stage 3c2e path for the classes above the fused caps, following the
// GEMM variant of the design doc. Per (li, lj, lk) class:
//
//   Phase A  H[ab_sph, (pp, p)]  bra E coefficients folded with the bra
//            cart2sph, geometry only; one block of columns per contracted
//            shell pair, aux independent, built once per class.
//   Phase B  X[(pp, p), col]     Boys + Hermite tower + geometry-free aux
//            fold + all scalar prefactors, accumulated over the primitives
//            of each contracted aux shell.
//   Phase C  V = H X per contracted pair (strided-batched DGEMM per run of
//            equal contraction degree, K = pair primitives x Hermite
//            index), then a scatter kernel writes V into the dense
//            [config, i_func, j_func, aux] result.
//
// Workspace layouts, all column major, all doubles, lab = li + lj. The K
// (contraction) dimension of contracted pair P stacks the Hermite index p
// over the pair primitives with the primitive fastest,
//   k = p * K_ab(P) + pp_local,   Ktot_P = nherm(lab) * K_ab(P),
// so that adjacent Phase B threads (adjacent primitive pairs) write
// adjacent X rows. Per-primitive-pair addressing arrays built once per
// class: kprefix[pp] = nherm(lab) * pair_prim_offsets[P] (the K prefix of
// P), ktot[pp] = Ktot_P, klocal[pp] = pp_local; passing NULL means every
// pair is its own contracted pair (K_ab = 1, used by the test entries).
//   H  one [M x Ktot_total] matrix per class, lda = M = nsph(li) nsph(lj);
//      column (P, pp_local, p) at kprefix + p * K_ab + pp_local.
//   X  per contracted pair P a [Ktot_P x N] matrix at kprefix * N,
//      ldb = Ktot_P; N = n_aux_shells_chunk * nsph(lk), column index
//      aux_shell_local * nsph(lk) + c_sph.
//   V  per contracted pair P a [M x N] tile at P * M * N, ldc = M.
//
// The total Hermite order is L = lab + lk + N_deriv with N_deriv = 0 for
// now; the tower and Boys depth are keyed on L so the derivative extension
// changes no interfaces.
namespace g3c::ts {

consteval int nsph(const int l) { return 2 * l + 1; }

// c2s and Cartesian exponent tables materialized as device constants so the
// table-driven kernels can index them at runtime
template <int l>
__device__ constexpr auto c2s_table = c2s_matrix<l>();
template <int l>
__device__ constexpr auto cart_table = cart_ls<l>();

// Phase A: one block per primitive pair. Threads 0..2 build the pair E
// coefficients per axis in shared memory, then the block sweeps the
// M x nherm(lab) matrix entries; each entry folds the bra cart2sph with the
// E coefficient product. Geometry only, no scalar prefactors (those live in
// X), so H is reusable across every aux group and chunk.
template <int li, int lj>
__global__ void h_matrix_kernel(double *h_matrix, const int *pair_prim_i,
                                const int *pair_prim_j,
                                const int n_prim_pairs,
                                const int *pp_kprefix, const int *pp_ktot,
                                const int *pp_klocal, const int *atm,
                                const int *bas, const double *env) {
  constexpr int lab = li + lj;
  constexpr int M = nsph(li) * nsph(lj);
  constexpr int n_herm = md::nherm(lab);

  const int pair_idx = blockIdx.x;
  if (pair_idx >= n_prim_pairs) return;
  const int kprefix = pp_kprefix ? pp_kprefix[pair_idx] : pair_idx * n_herm;
  const int k_ab = pp_ktot ? pp_ktot[pair_idx] / n_herm : 1;
  const int klocal = pp_klocal ? pp_klocal[pair_idx] : 0;

  __shared__ double e_x[md::nepair<li, lj>()];
  __shared__ double e_y[md::nepair<li, lj>()];
  __shared__ double e_z[md::nepair<li, lj>()];

  if (threadIdx.x < 3) {
    const int i_primitive = pair_prim_i[pair_idx];
    const int j_primitive = pair_prim_j[pair_idx];
    const double alpha = env[bas(PTR_EXP, i_primitive)];
    const double beta = env[bas(PTR_EXP, j_primitive)];
    const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_primitive));
    const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_primitive));
    const double pair_exponent = alpha + beta;
    const double a_weight = alpha / pair_exponent;
    const double b_weight = beta / pair_exponent;
    const double half_over_zeta = 0.5 / pair_exponent;
    const int axis = threadIdx.x;
    // P - A = w_b (B - A), P - B = -w_a (B - A)
    const double i_to_j =
        env[j_coord_offset + axis] - env[i_coord_offset + axis];
    if (axis == 0) {
      md::e_pair<li, lj>(e_x, b_weight * i_to_j, -a_weight * i_to_j,
                         half_over_zeta);
    } else if (axis == 1) {
      md::e_pair<li, lj>(e_y, b_weight * i_to_j, -a_weight * i_to_j,
                         half_over_zeta);
    } else {
      md::e_pair<li, lj>(e_z, b_weight * i_to_j, -a_weight * i_to_j,
                         half_over_zeta);
    }
  }
  __syncthreads();

  for (int out = threadIdx.x; out < M * n_herm; out += blockDim.x) {
    const int row = out % M; // ab_sph, rows are the fastest axis
    const int ioff = row / nsph(lj);
    const int joff = row % nsph(lj);
    const int p_index = out / M;
    const md::Triple p = md::herm_triple(p_index);

    double value = 0.0;
    for (int ic = 0; ic < ncart(li); ic++) {
      const double ci = c2s_table<li>[ioff][ic];
      if (ci <= 1e-15 && ci >= -1e-15) continue;
      const int ax = cart_table<li>[ic][0];
      const int ay = cart_table<li>[ic][1];
      const int az = cart_table<li>[ic][2];
      for (int jc = 0; jc < ncart(lj); jc++) {
        const double cj = c2s_table<lj>[joff][jc];
        if (cj <= 1e-15 && cj >= -1e-15) continue;
        const int bx = cart_table<lj>[jc][0];
        const int by = cart_table<lj>[jc][1];
        const int bz = cart_table<lj>[jc][2];
        if (p.t <= ax + bx && p.u <= ay + by && p.v <= az + bz) {
          value += ci * cj * e_x[md::eindex<li, lj>(ax, bx, p.t)] *
                   e_y[md::eindex<li, lj>(ay, by, p.u)] *
                   e_z[md::eindex<li, lj>(az, bz, p.v)];
        }
      }
    }
    const std::int64_t column = kprefix + p_index * k_ab + klocal;
    h_matrix[column * M + row] = value;
  }
}

// Phase A host launcher: instantiated for every li <= lj class in
// g3c2e_ts.cu (the table-driven kernels are cheap to compile, no sharding)
template <int li, int lj>
void launch_h_matrix(cudaStream_t stream, double *h_matrix,
                     const int *pair_prim_i, const int *pair_prim_j,
                     const int n_prim_pairs, const int *pp_kprefix,
                     const int *pp_ktot, const int *pp_klocal,
                     const int *atm, const int *bas, const double *env) {
  constexpr int work = nsph(li) * nsph(lj) * md::nherm(li + lj);
  constexpr int block_size = work < 256 ? (work + 31) / 32 * 32 : 256;
  h_matrix_kernel<li, lj><<<n_prim_pairs, block_size, 0, stream>>>(
      h_matrix, pair_prim_i, pair_prim_j, n_prim_pairs, pp_kprefix, pp_ktot,
      pp_klocal, atm, bas, env);
}

// Phase B (Mode R, L <= 8): one thread per (primitive pair, contracted aux
// shell). The Hermite tower lives in registers exactly like the fused
// kernel; the loop over the shell's aux primitives accumulates into the
// thread's X slice, so no atomics are needed. All scalar prefactors and the
// (-1)^lk ket parity are folded into X here; H stays geometry only.
template <int lab, int lk>
__global__ void hermite_aux_kernel(
    double *x_matrix, const int *pair_prim_i, const int *pair_prim_j,
    const int n_prim_pairs, const int *pp_kprefix, const int *pp_ktot,
    const int *pp_klocal, const int *aux_prim_indices,
    const int *aux_prim_offsets, const int n_cols, const int *atm,
    const int *bas, const double *env) {
  constexpr int L = lab + lk; // + N_deriv for the derivative extension
  constexpr int n_herm = md::nherm(lab);
  constexpr auto Ck = c2s_matrix<lk>();
  constexpr auto Bk = cart_ls<lk>();

  const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_idx >= n_prim_pairs) return;
  const int shell = blockIdx.y;

  const int i_primitive = pair_prim_i[pair_idx];
  const int j_primitive = pair_prim_j[pair_idx];
  const int ktot = pp_ktot ? pp_ktot[pair_idx] : n_herm;
  const int k_ab = ktot / n_herm;
  const int klocal = pp_klocal ? pp_klocal[pair_idx] : 0;
  const std::int64_t x_base =
      (std::int64_t)(pp_kprefix ? pp_kprefix[pair_idx]
                                : pair_idx * n_herm) *
      n_cols;

  const double alpha = env[bas(PTR_EXP, i_primitive)];
  const double beta = env[bas(PTR_EXP, j_primitive)];
  const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_primitive));
  const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_primitive));
  const double i_x = env[i_coord_offset + 0];
  const double i_y = env[i_coord_offset + 1];
  const double i_z = env[i_coord_offset + 2];
  const double ix_to_jx = env[j_coord_offset + 0] - i_x;
  const double iy_to_jy = env[j_coord_offset + 1] - i_y;
  const double iz_to_jz = env[j_coord_offset + 2] - i_z;
  const double pair_distance_squared =
      ix_to_jx * ix_to_jx + iy_to_jy * iy_to_jy + iz_to_jz * iz_to_jz;
  const double pair_exponent = alpha + beta;
  const double b_weight = beta / pair_exponent;
  const double p_x = i_x + b_weight * ix_to_jx;
  const double p_y = i_y + b_weight * iy_to_jy;
  const double p_z = i_z + b_weight * iz_to_jz;

  // no 0.5 on the primitive diagonal here: diagonal contracted pairs carry
  // the full primitive square and their V tile is halved in the scatter
  double pair_prefactor =
      env[bas(PTR_COEFF, i_primitive)] * env[bas(PTR_COEFF, j_primitive)] *
      exp(-alpha * beta / pair_exponent * pair_distance_squared);

  const int prim_begin = aux_prim_offsets[shell];
  const int prim_end = aux_prim_offsets[shell + 1];
  for (int aux_prim = prim_begin; aux_prim < prim_end; aux_prim++) {
    const int k_primitive = aux_prim_indices[aux_prim];
    const double gamma = env[bas(PTR_EXP, k_primitive)];
    const int k_coord_offset = atm(PTR_COORD, bas(ATOM_OF, k_primitive));
    const double w_x = p_x - env[k_coord_offset + 0];
    const double w_y = p_y - env[k_coord_offset + 1];
    const double w_z = p_z - env[k_coord_offset + 2];
    const double rho = pair_exponent * gamma / (pair_exponent + gamma);

    double f[L + 1];
    boys_function<L>(rho * (w_x * w_x + w_y * w_y + w_z * w_z), f);

    // [0]^(m) = prefactor * (-2 rho)^m F_m, prefactor as the fused kernel
    double prefactor = pair_prefactor * env[bas(PTR_COEFF, k_primitive)] *
                       34.98683665524972569 /
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

    double aux_power[lk + 1];
    aux_power[0] = 1.0;
    const double half_over_gamma = 0.5 / gamma;
    static_for<0, lk>([&]<int i>() {
      aux_power[i + 1] = aux_power[i] * half_over_gamma;
    });

    const bool first = aux_prim == prim_begin;
    static_for<0, nsph(lk)>([&]<int koff>() {
      double *x_column =
          x_matrix + x_base + (std::int64_t)(shell * nsph(lk) + koff) * ktot;

      static_for<0, n_herm>([&]<int ip>() {
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

        const int k_index = ip * k_ab + klocal;
        if (first) {
          x_column[k_index] = value;
        } else {
          x_column[k_index] += value;
        }
      });
    });
  }
}

// Precomputed Hermite index algebra shared by the warp kernel's tower and
// fold loops, sized for the largest two-stage tower L = 2 lab_max + lk_max
__device__ constexpr auto herm_table = md::make_herm_entries<md::nherm(
    2 * CUINT_G3C2E_TS_MAX_L_PAIR + CUINT_G3C2E_TS_MAX_L_AUX)>();

// Merged aux-fold table. In the fold, the aux_power exponent
// (cx+qx)/2 + (cy+qy)/2 + (cz+qz)/2 = (lk + |q|)/2 depends only on the
// Hermite offset q (cx+cy+cz = lk and the parities match per axis), so all
// Cartesians feeding the same q of one spherical component collapse into a
// single compile-time coefficient:
//   X(koff, p) = sum_q C[koff][q] aux_power[(lk+|q|)/2] r[hindex(p + q)]
//   C[koff][q] = sum_{c >= q, c = q (mod 2)} c2s(koff, c)
//                * aux_coef(cx, qx) aux_coef(cy, qy) aux_coef(cz, qz)
struct AuxFoldTerm {
  double coef;
  signed char qt, qu, qv, power;
};

template <int lk>
consteval double aux_fold_coef(const int koff, const int qx, const int qy,
                               const int qz) {
  const auto Ck = c2s_matrix<lk>();
  const auto Bk = cart_ls<lk>();
  double coef = 0.0;
  for (int kc = 0; kc < ncart(lk); kc++) {
    const int cx = Bk[kc][0], cy = Bk[kc][1], cz = Bk[kc][2];
    if (cx >= qx && cy >= qy && cz >= qz && (cx - qx) % 2 == 0 &&
        (cy - qy) % 2 == 0 && (cz - qz) % 2 == 0) {
      coef += Ck[koff][kc] * md::aux_coef(cx, qx) * md::aux_coef(cy, qy) *
              md::aux_coef(cz, qz);
    }
  }
  return coef;
}

template <int lk>
consteval int aux_fold_total() {
  int total = 0;
  for (int koff = 0; koff < nsph(lk); koff++)
    for (int qx = 0; qx <= lk; qx++)
      for (int qy = 0; qx + qy <= lk; qy++)
        for (int qz = 0; qx + qy + qz <= lk; qz++) {
          if ((qx + qy + qz) % 2 != lk % 2) continue;
          const double coef = aux_fold_coef<lk>(koff, qx, qy, qz);
          if (coef > 1e-15 || coef < -1e-15) total++;
        }
  return total;
}

template <int lk>
consteval std::array<int, nsph(lk) + 1> make_aux_fold_offsets() {
  std::array<int, nsph(lk) + 1> offsets{};
  int total = 0;
  for (int koff = 0; koff < nsph(lk); koff++) {
    offsets[koff] = total;
    for (int qx = 0; qx <= lk; qx++)
      for (int qy = 0; qx + qy <= lk; qy++)
        for (int qz = 0; qx + qy + qz <= lk; qz++) {
          if ((qx + qy + qz) % 2 != lk % 2) continue;
          const double coef = aux_fold_coef<lk>(koff, qx, qy, qz);
          if (coef > 1e-15 || coef < -1e-15) total++;
        }
  }
  offsets[nsph(lk)] = total;
  return offsets;
}

template <int lk>
consteval std::array<AuxFoldTerm, aux_fold_total<lk>()>
make_aux_fold_terms() {
  std::array<AuxFoldTerm, aux_fold_total<lk>()> terms{};
  int n = 0;
  for (int koff = 0; koff < nsph(lk); koff++)
    for (int qx = 0; qx <= lk; qx++)
      for (int qy = 0; qx + qy <= lk; qy++)
        for (int qz = 0; qx + qy + qz <= lk; qz++) {
          if ((qx + qy + qz) % 2 != lk % 2) continue;
          const double coef = aux_fold_coef<lk>(koff, qx, qy, qz);
          if (coef > 1e-15 || coef < -1e-15) {
            terms[n++] = {coef, (signed char)qx, (signed char)qy,
                          (signed char)qz,
                          (signed char)((lk + qx + qy + qz) / 2)};
          }
        }
  return terms;
}

template <int lk>
__device__ constexpr auto aux_fold_offsets = make_aux_fold_offsets<lk>();
template <int lk>
__device__ constexpr auto aux_fold_terms = make_aux_fold_terms<lk>();

// register-resident tower cap; above it Phase B switches to the
// shared-memory cooperative variant
constexpr int TsModeRLMax = 8;

// Phase B (Mode S, L > 8): one WARP per (primitive pair, contracted aux
// shell), several warps per block. The Hermite tower cannot be register
// resident (nherm(14) = 680 doubles), so each warp owns a shared-memory
// tower built cooperatively: the in-place downward recursion updates shells
// in descending order, entries within a shell are independent, so the 32
// lanes fill one shell in parallel with only a __syncwarp between shells
// (all control flow is warp-uniform, so full-mask syncs are valid). The aux
// fold then sweeps the nsph(lk) x nherm(lab) outputs using the merged
// compile-time fold table. Same X contract as Mode R.
constexpr int TsWarpsPerBlock = 4;

template <int lab, int lk>
__global__ void hermite_aux_kernel_warp(
    double *x_matrix, const int *pair_prim_i, const int *pair_prim_j,
    const int n_prim_pairs, const int *pp_kprefix, const int *pp_ktot,
    const int *pp_klocal, const int *aux_prim_indices,
    const int *aux_prim_offsets, const int n_cols, const int *atm,
    const int *bas, const double *env) {
  constexpr int L = lab + lk; // + N_deriv for the derivative extension
  constexpr int n_herm = md::nherm(lab);

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair_idx = blockIdx.x * TsWarpsPerBlock + warp;
  if (pair_idx >= n_prim_pairs) return; // warp-uniform exit
  const int shell = blockIdx.y;

  const int i_primitive = pair_prim_i[pair_idx];
  const int j_primitive = pair_prim_j[pair_idx];
  const int ktot = pp_ktot ? pp_ktot[pair_idx] : n_herm;
  const int k_ab = ktot / n_herm;
  const int klocal = pp_klocal ? pp_klocal[pair_idx] : 0;
  const std::int64_t x_base =
      (std::int64_t)(pp_kprefix ? pp_kprefix[pair_idx]
                                : pair_idx * n_herm) *
      n_cols;

  const double alpha = env[bas(PTR_EXP, i_primitive)];
  const double beta = env[bas(PTR_EXP, j_primitive)];
  const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_primitive));
  const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_primitive));
  const double i_x = env[i_coord_offset + 0];
  const double i_y = env[i_coord_offset + 1];
  const double i_z = env[i_coord_offset + 2];
  const double ix_to_jx = env[j_coord_offset + 0] - i_x;
  const double iy_to_jy = env[j_coord_offset + 1] - i_y;
  const double iz_to_jz = env[j_coord_offset + 2] - i_z;
  const double pair_distance_squared =
      ix_to_jx * ix_to_jx + iy_to_jy * iy_to_jy + iz_to_jz * iz_to_jz;
  const double pair_exponent = alpha + beta;
  const double b_weight = beta / pair_exponent;
  const double p_x = i_x + b_weight * ix_to_jx;
  const double p_y = i_y + b_weight * iy_to_jy;
  const double p_z = i_z + b_weight * iz_to_jz;
  const double pair_prefactor =
      env[bas(PTR_COEFF, i_primitive)] * env[bas(PTR_COEFF, j_primitive)] *
      exp(-alpha * beta / pair_exponent * pair_distance_squared);

  __shared__ double r_all[TsWarpsPerBlock][md::nherm(L)];
  __shared__ double f_all[TsWarpsPerBlock][L + 1];
  __shared__ double pw_all[TsWarpsPerBlock][lk + 1];
  double *const r = r_all[warp];
  double *const f = f_all[warp];
  double *const aux_power = pw_all[warp];

  const int prim_begin = aux_prim_offsets[shell];
  const int prim_end = aux_prim_offsets[shell + 1];
  for (int aux_prim = prim_begin; aux_prim < prim_end; aux_prim++) {
    const int k_primitive = aux_prim_indices[aux_prim];
    const double gamma = env[bas(PTR_EXP, k_primitive)];
    const int k_coord_offset = atm(PTR_COORD, bas(ATOM_OF, k_primitive));
    const double w_x = p_x - env[k_coord_offset + 0];
    const double w_y = p_y - env[k_coord_offset + 1];
    const double w_z = p_z - env[k_coord_offset + 2];
    const double rho = pair_exponent * gamma / (pair_exponent + gamma);

    if (lane == 0) {
      double f_local[L + 1];
      boys_function<L>(rho * (w_x * w_x + w_y * w_y + w_z * w_z), f_local);
      double prefactor = pair_prefactor *
                         env[bas(PTR_COEFF, k_primitive)] *
                         34.98683665524972569 /
                         (pair_exponent * gamma *
                          sqrt(pair_exponent + gamma));
      if constexpr (lk % 2 == 1) {
        prefactor = -prefactor;
      }
      double scale = prefactor;
      static_for<0, L + 1>([&]<int m>() {
        f[m] = f_local[m] * scale;
        scale *= -2.0 * rho;
      });

      aux_power[0] = 1.0;
      const double half_over_gamma = 0.5 / gamma;
      static_for<0, lk>([&]<int i>() {
        aux_power[i + 1] = aux_power[i] * half_over_gamma;
      });

      r[0] = f[L]; // [0]^(L)
    }
    __syncwarp();

    // cooperative in-place tower: shells descend, entries within a shell
    // are independent (mirrors md::hermite_tower)
    for (int step = 0; step < L; step++) {
      const int m = L - 1 - step;
      for (int s = L - m; s >= 1; s--) {
        for (int i = md::nherm(s - 1) + lane; i < md::nherm(s);
             i += 32) {
          const md::HermEntry e = herm_table[i];
          const double w = e.axis == 0 ? w_x : (e.axis == 1 ? w_y : w_z);
          double value = w * r[e.low1];
          if (e.comp) {
            value += (double)e.comp * r[e.low2];
          }
          r[i] = value;
        }
        __syncwarp();
      }
      if (lane == 0) {
        r[0] = f[m];
      }
      __syncwarp();
    }

    // aux fold over the (c_sph, p) outputs: koff is warp-uniform so the
    // fold-table loads broadcast; lanes stride the Hermite index
    const bool first = aux_prim == prim_begin;
    for (int koff = 0; koff < nsph(lk); koff++) {
      double *x_column =
          x_matrix + x_base +
          (std::int64_t)(shell * nsph(lk) + koff) * ktot;
      const int term_begin = aux_fold_offsets<lk>[koff];
      const int term_end = aux_fold_offsets<lk>[koff + 1];
      for (int ip = lane; ip < n_herm; ip += 32) {
        const md::HermEntry p = herm_table[ip];

        double value = 0.0;
        for (int t = term_begin; t < term_end; t++) {
          const AuxFoldTerm term = aux_fold_terms<lk>[t];
          value += term.coef * aux_power[term.power] *
                   r[md::hindex(p.t + term.qt, p.u + term.qu,
                                p.v + term.qv)];
        }

        const int k_index = ip * k_ab + klocal;
        if (first) {
          x_column[k_index] = value;
        } else {
          x_column[k_index] += value;
        }
      }
    }
    __syncwarp(); // the next primitive overwrites the shared tower
  }
}

// Phase B host launcher: instantiated per (lab, lk) in the per-lk shards
template <int lab, int lk>
void launch_hermite_aux(cudaStream_t stream, double *x_matrix,
                        const int *pair_prim_i, const int *pair_prim_j,
                        const int n_prim_pairs, const int *pp_kprefix,
                        const int *pp_ktot, const int *pp_klocal,
                        const int *aux_prim_indices,
                        const int *aux_prim_offsets, const int n_aux_shells,
                        const int *atm, const int *bas, const double *env) {
  const int n_cols = n_aux_shells * nsph(lk);
  if constexpr (lab + lk <= TsModeRLMax) {
    const dim3 block_size{256, 1, 1};
    const dim3 block_grid{(uint)((n_prim_pairs + 255) / 256),
                          (uint)n_aux_shells, 1};
    hermite_aux_kernel<lab, lk><<<block_grid, block_size, 0, stream>>>(
        x_matrix, pair_prim_i, pair_prim_j, n_prim_pairs, pp_kprefix,
        pp_ktot, pp_klocal, aux_prim_indices, aux_prim_offsets, n_cols, atm,
        bas, env);
  } else {
    const dim3 block_size{32 * TsWarpsPerBlock, 1, 1};
    const dim3 block_grid{
        (uint)((n_prim_pairs + TsWarpsPerBlock - 1) / TsWarpsPerBlock),
        (uint)n_aux_shells, 1};
    hermite_aux_kernel_warp<lab, lk><<<block_grid, block_size, 0, stream>>>(
        x_matrix, pair_prim_i, pair_prim_j, n_prim_pairs, pp_kprefix,
        pp_ktot, pp_klocal, aux_prim_indices, aux_prim_offsets, n_cols, atm,
        bas, env);
  }
}

// signature of launch_hermite_aux<lab, lk>, shared by the extern template
// declarations and the per-lk explicit instantiation shards
#define G3C2E_TS_LAUNCH_B_PARAMS                                             \
  (cudaStream_t, double *, const int *, const int *, const int,             \
   const int *, const int *, const int *, const int *, const int *,         \
   const int, const int *, const int *, const double *)

// runtime dispatchers over the launcher templates, defined in g3c2e_ts.cu;
// used by the phase loop there and by the testing entries in
// g3c2e_ts_test.cu
void dispatch_h_matrix(cudaStream_t stream, double *h_matrix,
                          const int *pair_prim_i, const int *pair_prim_j,
                          const int n_prim_pairs, const int *pp_kprefix,
                          const int *pp_ktot, const int *pp_klocal,
                          const int *atm, const int *bas, const double *env,
                          const int i_angular, const int j_angular);
void dispatch_hermite_aux(
    cudaStream_t stream, double *x_matrix, const int *pair_prim_i,
    const int *pair_prim_j, const int n_prim_pairs, const int *pp_kprefix,
    const int *pp_ktot, const int *pp_klocal, const int *aux_prim_indices,
    const int *aux_prim_offsets, const int n_aux_shells, const int *atm,
    const int *bas, const double *env, const int lab, const int k_angular);

}  // namespace g3c::ts
