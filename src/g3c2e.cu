#include "g3c2e.cuh"

namespace g3c {

// instantiated in the per-lk g3c2e_ints.cu shards
extern template void launch<0> G3C2E_LAUNCH_PARAMS;
#if CUINT_G3C2E_MAX_L_AUX >= 1
extern template void launch<1> G3C2E_LAUNCH_PARAMS;
#endif
#if CUINT_G3C2E_MAX_L_AUX >= 2
extern template void launch<2> G3C2E_LAUNCH_PARAMS;
#endif
#if CUINT_G3C2E_MAX_L_AUX >= 3
extern template void launch<3> G3C2E_LAUNCH_PARAMS;
#endif
#if CUINT_G3C2E_MAX_L_AUX >= 4
extern template void launch<4> G3C2E_LAUNCH_PARAMS;
#endif
#if CUINT_G3C2E_MAX_L_AUX >= 5
extern template void launch<5> G3C2E_LAUNCH_PARAMS;
#endif
#if CUINT_G3C2E_MAX_L_AUX >= 6
extern template void launch<6> G3C2E_LAUNCH_PARAMS;
#endif
#if CUINT_G3C2E_MAX_L_AUX >= 7
#error "add extern template declarations for G3C2E_MAX_L_AUX > 6"
#endif

}  // namespace g3c

void int3c2e_fused(cudaStream_t stream, double *result, const int *pair_indices,
                   const int n_pairs, const int n_primitives,
                   const int *primitive_to_function, const int n_functions,
                   const int *aux_indices, const int n_aux_primitives,
                   const int *aux_primitive_to_function, const int n_aux,
                   const int *atm, const int atm_stride, const int *bas,
                   const int bas_stride, const double *env, const int env_stride,
                   const int n_configurations, const int i_angular,
                   const int j_angular, const int k_angular, const int is_screened) {
  dispatch_range<0, G3cAuxLMAX1>(k_angular, [&]<int lk>() {
    g3c::launch<lk>(stream, result, pair_indices, n_pairs, n_primitives,
                    primitive_to_function, n_functions, aux_indices,
                    n_aux_primitives, aux_primitive_to_function, n_aux, atm,
                    atm_stride, bas, bas_stride, env, env_stride,
                    n_configurations, i_angular, j_angular, is_screened);
  });
}
