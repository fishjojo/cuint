#pragma once

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

#ifdef __cplusplus
}
#endif
