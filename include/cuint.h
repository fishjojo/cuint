#pragma once

#include <stddef.h>

#define PTR_COMMON_ORIG    1

#ifdef __cplusplus
extern "C" {
#endif

void overlap(cudaStream_t stream,
             double *result, const int *pair_indices, const int n_pairs,
             const int n_primitives, const int *primitive_to_function,
             const int n_functions, const int *atm, const int atm_stride,
             const int *bas, const int bas_stride, const double *env,
             const int env_stride, const int n_configurations,
             const int i_angular, const int j_angular, const int is_screened);

void overlap_gradient(cudaStream_t stream,
                      double *result, const int *pair_indices,
                      const int n_pairs, const int n_primitives,
                      const int *primitive_to_function, const int n_functions,
                      const int *atm, const int atm_stride, const int *bas,
                      const int bas_stride, const double *env,
                      const int env_stride, const int n_configurations,
                      const int i_angular, const int j_angular,
                      const int is_screened);

void gen_overlap(cudaStream_t stream,
                 double *result, const int *pair_indices, const int n_pairs,
                 const int n_primitives, const int *primitive_to_function,
                 const int n_functions, const int *atm, const int atm_stride,
                 const int *bas, const int bas_stride, const double *env,
                 const int env_stride, const int n_configurations,
                 const int i_angular, const int j_angular,
                 const int is_screened, const int i_deriv, const int j_deriv,
                 const int comp);

void dipole(cudaStream_t stream,
            double *result, const int *pair_indices, const int n_pairs,
            const int n_primitives, const int *primitive_to_function,
            const int n_functions, const int *atm, const int atm_stride,
            const int *bas, const int bas_stride, const double *env,
            const int env_stride, const int n_configurations,
            const int i_angular, const int j_angular,
            const int is_screened);

void dipole_gradient(cudaStream_t stream,
                     double *result, const int *pair_indices, const int n_pairs,
                     const int n_primitives, const int *primitive_to_function,
                     const int n_functions, const int *atm,
                     const int atm_stride, const int *bas, const int bas_stride,
                     const double *env, const int env_stride,
                     const int n_configurations, const int i_angular,
                     const int j_angular, const int is_screened);

void quadrupole(cudaStream_t stream,
                double *result, const int *pair_indices, const int n_pairs,
                const int n_primitives, const int *primitive_to_function,
                const int n_functions, const int *atm, const int atm_stride,
                const int *bas, const int bas_stride, const double *env,
                const int env_stride, const int n_configurations,
                const int i_angular, const int j_angular,
                const int is_screened);

void quadrupole_gradient(
    cudaStream_t stream,
    double *result, const int *pair_indices, const int n_pairs,
    const int n_primitives, const int *primitive_to_function,
    const int n_functions, const int *atm, const int atm_stride, const int *bas,
    const int bas_stride, const double *env, const int env_stride,
    const int n_configurations, const int i_angular, const int j_angular,
    const int is_screened);

// Optional plan-time metadata for the two-stage path of int3c2e. When the
// caller stages this once (see cuint/int3c2e.py), the two-stage dispatch
// issues launches only -- no device-to-host copies, no synchronization --
// so classes on distinct streams overlap. The blobs are shared across
// classes: pair_meta is built once per (i_angular, j_angular) pair class
// and aux_meta once per k_angular group, while the aux primitive arrays
// are the ordinary aux_indices / aux_primitive_to_function arguments of
// int3c2e (no duplication). Layouts (device int32):
//   pair_meta  pair_prim_i | pair_prim_j | kprefix | ktot | klocal |
//              pair_i_function | pair_j_function   (contracted pairs
//              I <= J sorted by contraction degree)
//   aux_meta   aux_prim_offsets (n_aux_shells + 1) | aux_function
// run_* are HOST descriptors of the contiguous equal-contraction-degree
// spans of the pair list (first pair, pair count, K_ab,
// pair_prim_offsets[first]); they depend only on the pair class.
// chunk_shells is the aux chunk size of the [H][X chunk][V chunk]
// workspace partition, computed at plan time against the workspace slot
// size. build_h == 0 skips the Phase A rebuild when H for the same pair
// class is still in the workspace from the previous call on the SAME
// stream (single configuration only; ignored otherwise).
typedef struct Int3c2eTsPlan {
  const int *pair_meta;
  const int *aux_meta;
  const int *run_first;
  const int *run_count;
  const int *run_k_ab;
  const int *run_offset;
  int n_prim_pairs;
  int n_contracted_pairs;
  int n_aux_shells;
  int n_runs;
  int chunk_shells;
  int build_h;
} Int3c2eTsPlan;

// Unified 3c2e entry point for one (i_angular, j_angular, k_angular) class.
// The arguments are those of the fused path (device pointers throughout)
// plus a device workspace: a class within the effective fused caps runs the
// fused single-kernel path (ts_plan and workspace ignored, may be NULL),
// every other class runs the two-stage path -- from the plan-time metadata
// when ts_plan is given (no host synchronization), otherwise by deriving
// the contracted shell structure from the runs of equal values in
// primitive_to_function / aux_primitive_to_function (blocking). Both
// two-stage modes need the workspace. Screening is only supported within
// the fused caps. Returns 0 on success.
int int3c2e(cudaStream_t stream,
            double *result, const int *pair_indices, const int n_pairs,
            const int n_primitives, const int *primitive_to_function,
            const int n_functions, const int *aux_indices,
            const int n_aux_primitives,
            const int *aux_primitive_to_function, const int n_aux,
            const int *atm, const int atm_stride, const int *bas,
            const int bas_stride, const double *env, const int env_stride,
            const int n_configurations, const int i_angular,
            const int j_angular, const int k_angular, const int is_screened,
            const Int3c2eTsPlan *ts_plan, double *workspace,
            const size_t workspace_bytes);

// Effective fused caps used by the int3c2e dispatch (per-index pair/aux and
// total li + lj + lk), clamped to the compiled ones. Lowering them (e.g. to
// -1) forces classes through the two-stage path; useful for testing and for
// tuning the crossover.
void int3c2e_set_fused_caps(const int max_l_pair, const int max_l_aux,
                            const int max_l_total);

// Fused single-kernel path for one class within the fused caps. Direct
// (benchmarking) entry; int3c2e above is the primary API.
void int3c2e_fused(cudaStream_t stream,
             double *result, const int *pair_indices, const int n_pairs,
             const int n_primitives, const int *primitive_to_function,
             const int n_functions, const int *aux_indices,
             const int n_aux_primitives,
             const int *aux_primitive_to_function, const int n_aux,
             const int *atm, const int atm_stride, const int *bas,
             const int bas_stride, const double *env, const int env_stride,
             const int n_configurations, const int i_angular,
             const int j_angular, const int k_angular, const int is_screened);

// Two-stage path driven by plan-time metadata (see Int3c2eTsPlan above);
// aux_prim_indices is the same device array int3c2e receives as
// aux_indices, and workspace is partitioned into [H][X chunk][V chunk]
// using ts_plan->chunk_shells. Building block of the ts_plan branch;
// int3c2e above is the primary API. Returns 0 on success.
int int3c2e_two_stage_planned(
    cudaStream_t stream, double *result, const Int3c2eTsPlan *ts_plan,
    const int *aux_prim_indices, const int n_aux_prims,
    const int n_functions, const int n_aux, const int *atm,
    const int atm_stride, const int *bas, const int bas_stride,
    const double *env, const int env_stride, const int n_configurations,
    const int i_angular, const int j_angular, const int k_angular,
    double *workspace);

// In-place symmetrization of the dense (config, i, j, aux) result produced
// by the int3c2e classes: both (i, j) and (j, i) tiles become their sum,
// equivalent to result + result.transpose(0, 2, 1, 3) without allocating a
// second tensor. result is a device pointer.
void int3c2e_symmetrize(cudaStream_t stream, double *result,
                        const int n_configurations, const int n_functions,
                        const int n_aux);

void pbc_overlap(cudaStream_t stream,
                 double *result, const int *pair_indices, const int n_pairs,
                 const int n_primitives, const int *primitive_to_function,
                 const int n_functions, const int *atm, const int atm_stride,
                 const int *bas, const int bas_stride, const double *env,
                 const int env_stride, const int n_configurations,
                 const double *Ls, const int n_images, const int *mask,
                 const int i_angular, const int j_angular,
                 const int is_screened, const int reduce_over_images);

void pbc_overlap_gradient(cudaStream_t stream,
                          double *result, const int *pair_indices,
                          const int n_pairs, const int n_primitives,
                          const int *primitive_to_function, const int n_functions,
                          const int *atm, const int atm_stride, const int *bas,
                          const int bas_stride, const double *env,
                          const int env_stride, const int n_configurations,
                          const double *Ls, const int n_images, const int *mask,
                          const int i_angular, const int j_angular,
                          const int is_screened, const int reduce_over_images);

#ifdef __cplusplus
}
#endif
