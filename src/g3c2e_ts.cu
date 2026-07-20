#include <algorithm>
#include <cstdio>
#include <map>
#include <vector>

#include <cublas_v2.h>

#include "g3c2e_ts.cuh"

namespace g3c::ts {

// Phase B launchers are instantiated in the per-lk g3c2e_ts_ints.cu shards.
// Declarations above the compiled caps are never ODR-used because every
// dispatch below is bounded by the same config macros.
#define G3C2E_TS_EXTERN_LK(LK)                                            \
  extern template void launch_hermite_aux<0, LK>                         \
      G3C2E_TS_LAUNCH_B_PARAMS;                                          \
  extern template void launch_hermite_aux<1, LK>                         \
      G3C2E_TS_LAUNCH_B_PARAMS;                                          \
  extern template void launch_hermite_aux<2, LK>                         \
      G3C2E_TS_LAUNCH_B_PARAMS;                                          \
  extern template void launch_hermite_aux<3, LK>                         \
      G3C2E_TS_LAUNCH_B_PARAMS;                                          \
  extern template void launch_hermite_aux<4, LK>                         \
      G3C2E_TS_LAUNCH_B_PARAMS;                                          \
  extern template void launch_hermite_aux<5, LK>                         \
      G3C2E_TS_LAUNCH_B_PARAMS;                                          \
  extern template void launch_hermite_aux<6, LK>                         \
      G3C2E_TS_LAUNCH_B_PARAMS;                                          \
  extern template void launch_hermite_aux<7, LK>                         \
      G3C2E_TS_LAUNCH_B_PARAMS;                                          \
  extern template void launch_hermite_aux<8, LK>                         \
      G3C2E_TS_LAUNCH_B_PARAMS;

G3C2E_TS_EXTERN_LK(0)
#if CUINT_G3C2E_TS_MAX_L_AUX >= 1
G3C2E_TS_EXTERN_LK(1)
#endif
#if CUINT_G3C2E_TS_MAX_L_AUX >= 2
G3C2E_TS_EXTERN_LK(2)
#endif
#if CUINT_G3C2E_TS_MAX_L_AUX >= 3
G3C2E_TS_EXTERN_LK(3)
#endif
#if CUINT_G3C2E_TS_MAX_L_AUX >= 4
G3C2E_TS_EXTERN_LK(4)
#endif
#if CUINT_G3C2E_TS_MAX_L_AUX >= 5
G3C2E_TS_EXTERN_LK(5)
#endif
#if CUINT_G3C2E_TS_MAX_L_AUX >= 6
G3C2E_TS_EXTERN_LK(6)
#endif
#if CUINT_G3C2E_TS_MAX_L_AUX >= 7
#error "add extern template declarations for G3C2E_TS_MAX_L_AUX > 6"
#endif

// dispatch a (li <= lj) pair class to the Phase A launcher (declared in
// g3c2e_ts.cuh; also used by the testing entries in g3c2e_ts_test.cu)
void dispatch_h_matrix(cudaStream_t stream, double *h_matrix,
                       const int *pair_prim_i, const int *pair_prim_j,
                       const int n_prim_pairs, const int *pp_kprefix,
                       const int *pp_ktot, const int *pp_klocal,
                       const int *atm, const int *bas, const double *env,
                       const int i_angular, const int j_angular) {
  dispatch_range<0, G3cTsPairLMAX1>(i_angular, [&]<int li>() {
    dispatch_range<li, G3cTsPairLMAX1>(j_angular, [&]<int lj>() {
      launch_h_matrix<li, lj>(stream, h_matrix, pair_prim_i, pair_prim_j,
                              n_prim_pairs, pp_kprefix, pp_ktot, pp_klocal,
                              atm, bas, env);
    });
  });
}

// dispatch (lab, lk) to the Phase B launcher (declared in g3c2e_ts.cuh;
// also used by the testing entries in g3c2e_ts_test.cu)
void dispatch_hermite_aux(
    cudaStream_t stream, double *x_matrix, const int *pair_prim_i,
    const int *pair_prim_j, const int n_prim_pairs, const int *pp_kprefix,
    const int *pp_ktot, const int *pp_klocal, const int *aux_prim_indices,
    const int *aux_prim_offsets, const int n_aux_shells, const int *atm,
    const int *bas, const double *env, const int lab, const int k_angular) {
  dispatch_range<0, 2 * CUINT_G3C2E_TS_MAX_L_PAIR + 1>(lab, [&]<int LAB>() {
    dispatch_range<0, G3cTsAuxLMAX1>(k_angular, [&]<int lk>() {
      launch_hermite_aux<LAB, lk>(stream, x_matrix, pair_prim_i, pair_prim_j,
                                  n_prim_pairs, pp_kprefix, pp_ktot,
                                  pp_klocal, aux_prim_indices,
                                  aux_prim_offsets, n_aux_shells, atm, bas,
                                  env);
    });
  });
}

// Phase C scatter: add the V tiles into the dense result. Tiles are
// disjoint across contracted pairs, aux chunks, and classes, so plain
// read-modify-write suffices. Diagonal (I == J) tiles carry the full
// primitive square and are halved so the host-side transpose symmetrization
// reconstructs them exactly.
static __global__ void scatter_tiles(
    double *result, const double *v_tiles, const int *pair_i_function,
    const int *pair_j_function, const int *aux_function,
    const std::int64_t n_elements, const int nsph_i, const int nsph_j,
    const int nsph_k, const int n_cols, const int n_functions,
    const int n_aux, const int halve_diagonal) {
  const int M = nsph_i * nsph_j;
  for (std::int64_t idx =
           (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < n_elements; idx += (std::int64_t)gridDim.x * blockDim.x) {
    const int pair = idx / ((std::int64_t)M * n_cols);
    const int rem = idx - (std::int64_t)pair * M * n_cols;
    const int col = rem / M;
    const int row = rem - col * M;

    double value = v_tiles[idx];
    const int i_function = pair_i_function[pair];
    const int j_function = pair_j_function[pair];
    if (halve_diagonal && i_function == j_function) {
      value *= 0.5;
    }
    const int i_func = i_function + row / nsph_j;
    const int j_func = j_function + row % nsph_j;
    const int k_func = aux_function[col / nsph_k] + col % nsph_k;
    result[((std::int64_t)i_func * n_functions + j_func) * n_aux + k_func] +=
        value;
  }
}

// one cuBLAS handle per stream: concurrently executing streams must not
// share a handle (its internal workspace would race), and binding the
// stream once at creation keeps the hot path free of cublasSetStream calls.
// The host loop is single threaded, so no locking.
static cublasHandle_t handle_for_stream(cudaStream_t stream) {
  static std::map<cudaStream_t, cublasHandle_t> handles;
  if (!handles.contains(stream)) {
    cublasHandle_t h;
    cublasCreate(&h);
    cublasSetStream(h, stream);
    handles[stream] = h;
  }
  return handles[stream];
}

}  // namespace g3c::ts

namespace {

// one strided-batched GEMM per contiguous run of equal contraction degree
struct TsGemmRun {
  int first, count, k_ab, offset; // offset = pair_prim_offsets[first]
};

// Shared phase loop: everything it consumes is either already on the
// device (the metadata arrays, the [H][X][V] workspace) or host-side run
// descriptors, so it issues launches only -- no copies, no
// synchronization. build_h == 0 skips the Phase A rebuild (valid only when
// H from a previous class with the same (li, lj) is still in the
// workspace and n_configurations == 1; enforced by the callers).
int two_stage_core(
    cudaStream_t stream, double *result, const int *d_pair_prim_i,
    const int *d_pair_prim_j, const int *d_kprefix, const int *d_ktot,
    const int *d_klocal, const int *d_pair_i_function,
    const int *d_pair_j_function, const int *d_aux_prims,
    const int *d_aux_offsets, const int *d_aux_function,
    const int n_prim_pairs, const int n_contracted_pairs,
    const int n_aux_shells, const TsGemmRun *runs, const int n_runs,
    const int chunk_shells, const int build_h, const int n_functions,
    const int n_aux, const int *atm, const int atm_stride, const int *bas,
    const int bas_stride, const double *env, const int env_stride,
    const int n_configurations, const int i_angular, const int j_angular,
    const int k_angular, double *h_matrix) {
  using std::int64_t;

  const int lab = i_angular + j_angular;
  const int n_herm = md::nherm(lab);
  const int M = (2 * i_angular + 1) * (2 * j_angular + 1);
  const int nsph_k = 2 * k_angular + 1;
  const int64_t ktot_total = (int64_t)n_herm * n_prim_pairs;
  double *x_matrix = h_matrix + (int64_t)M * ktot_total;
  double *v_tiles = x_matrix + ktot_total * chunk_shells * nsph_k;

  cublasHandle_t handle = g3c::ts::handle_for_stream(stream);
  const int halve_diagonal = i_angular == j_angular;

  for (int config = 0; config < n_configurations; config++) {
    const int *atm_c = atm + (int64_t)config * atm_stride;
    const int *bas_c = bas + (int64_t)config * bas_stride;
    const double *env_c = env + (int64_t)config * env_stride;
    double *result_c =
        result + (int64_t)config * n_functions * n_functions * n_aux;

    if (config > 0 || build_h) {
      g3c::ts::dispatch_h_matrix(stream, h_matrix, d_pair_prim_i,
                                 d_pair_prim_j, n_prim_pairs, d_kprefix,
                                 d_ktot, d_klocal, atm_c, bas_c, env_c,
                                 i_angular, j_angular);
    }

    for (int shell_begin = 0; shell_begin < n_aux_shells;
         shell_begin += chunk_shells) {
      const int shells =
          std::min(chunk_shells, n_aux_shells - shell_begin);
      const int n_cols = shells * nsph_k;

      g3c::ts::dispatch_hermite_aux(
          stream, x_matrix, d_pair_prim_i, d_pair_prim_j, n_prim_pairs,
          d_kprefix, d_ktot, d_klocal, d_aux_prims,
          d_aux_offsets + shell_begin, shells, atm_c, bas_c, env_c, lab,
          k_angular);

      const double one = 1.0, zero = 0.0;
      for (int r = 0; r < n_runs; r++) {
        const TsGemmRun &run = runs[r];
        const int k = n_herm * run.k_ab;
        const int64_t offset = run.offset;
        const cublasStatus_t status = cublasDgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, M, n_cols, k, &one,
            h_matrix + n_herm * offset * M, M, (int64_t)k * M,
            x_matrix + n_herm * offset * n_cols, k, (int64_t)k * n_cols,
            &zero, v_tiles + (int64_t)run.first * M * n_cols, M,
            (int64_t)M * n_cols, run.count);
        if (status != CUBLAS_STATUS_SUCCESS) {
          std::fprintf(stderr,
                       "int3c2e_two_stage: batched DGEMM failed (%d)\n",
                       (int)status);
          return 5;
        }
      }

      const int64_t n_scatter = (int64_t)n_contracted_pairs * M * n_cols;
      const int scatter_blocks =
          (int)std::min<int64_t>((n_scatter + 255) / 256, 1 << 20);
      g3c::ts::scatter_tiles<<<scatter_blocks, 256, 0, stream>>>(
          result_c, v_tiles, d_pair_i_function, d_pair_j_function,
          d_aux_function + shell_begin, n_scatter, 2 * i_angular + 1,
          2 * j_angular + 1, nsph_k, n_cols, n_functions, n_aux,
          halve_diagonal);
    }
  }

  if (cudaGetLastError() != cudaSuccess) {
    std::fprintf(stderr, "int3c2e_two_stage: kernel launch failed\n");
    return 6;
  }
  return 0;
}

}  // namespace

int int3c2e_two_stage_planned(
    cudaStream_t stream, double *result, const Int3c2eTsPlan *ts_plan,
    const int *aux_prim_indices, const int n_aux_prims,
    const int n_functions, const int n_aux, const int *atm,
    const int atm_stride, const int *bas, const int bas_stride,
    const double *env, const int env_stride, const int n_configurations,
    const int i_angular, const int j_angular, const int k_angular,
    double *workspace) {
  const int n_contracted_pairs = ts_plan->n_contracted_pairs;
  const int n_aux_shells = ts_plan->n_aux_shells;
  if (n_contracted_pairs == 0 || n_aux_shells == 0) {
    return 0;
  }
  // the chunk size was computed at plan time against the workspace slot
  if (ts_plan->chunk_shells < 1) {
    std::fprintf(stderr,
                 "int3c2e_two_stage_planned: invalid chunk_shells (%d)\n",
                 ts_plan->chunk_shells);
    return 3;
  }

  std::vector<TsGemmRun> runs(ts_plan->n_runs);
  for (int r = 0; r < ts_plan->n_runs; r++) {
    runs[r] = {ts_plan->run_first[r], ts_plan->run_count[r],
               ts_plan->run_k_ab[r], ts_plan->run_offset[r]};
  }

  const int *d_pair_prim_i = ts_plan->pair_meta;
  const int *d_pair_prim_j = d_pair_prim_i + ts_plan->n_prim_pairs;
  const int *d_kprefix = d_pair_prim_j + ts_plan->n_prim_pairs;
  const int *d_ktot = d_kprefix + ts_plan->n_prim_pairs;
  const int *d_klocal = d_ktot + ts_plan->n_prim_pairs;
  const int *d_pair_i_function = d_klocal + ts_plan->n_prim_pairs;
  const int *d_pair_j_function = d_pair_i_function + n_contracted_pairs;
  const int *d_aux_offsets = ts_plan->aux_meta;
  const int *d_aux_function = d_aux_offsets + n_aux_shells + 1;

  // the Phase A reuse contract only holds within one configuration
  const int build_h = n_configurations > 1 ? 1 : ts_plan->build_h;
  return two_stage_core(stream, result, d_pair_prim_i, d_pair_prim_j,
                        d_kprefix, d_ktot, d_klocal, d_pair_i_function,
                        d_pair_j_function, aux_prim_indices, d_aux_offsets,
                        d_aux_function, ts_plan->n_prim_pairs,
                        n_contracted_pairs, n_aux_shells, runs.data(),
                        ts_plan->n_runs, ts_plan->chunk_shells, build_h,
                        n_functions, n_aux, atm, atm_stride, bas,
                        bas_stride, env, env_stride, n_configurations,
                        i_angular, j_angular, k_angular, workspace);
}
