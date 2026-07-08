#pragma once

#include "utils.h"

namespace rr {

template <int angular>
__forceinline__ __device__ constexpr double common_fac_sp() {
  if constexpr (angular == 0) {
    return 0.282094791773878143;
  } else if constexpr (angular == 1) {
    return 0.488602511902919921;
  } else {
    return 1.0;
  }
}

template <int angular>
__forceinline__ __device__ void
vertical_recursion(double result[], const double a00,
                   const double factor_from_previous,
                   const double factor_from_second_previous) {
  result[0] = a00;
  if constexpr (angular > 0) {
    result[1] = factor_from_previous * a00;
  }

  static_for<1, angular>([&](auto i_) {
    constexpr int i = decltype(i_)::value;
    result[i + 1] = i * factor_from_second_previous * result[i - 1] +
                    factor_from_previous * result[i];
  });
}

template <int i_angular, int j_angular, int stride>
__forceinline__ __device__ void insert_position_operator(double result[],
                                                         const double shift) {
#pragma unroll
  for (int i = 0; i <= i_angular; i++) {
#pragma unroll
    for (int j = 0; j <= j_angular; j++) {
      result[i * stride + j] =
          result[i * stride + j + 1] + shift * result[i * stride + j];
    }
  }
}

template <int i_angular, int j_angular, int stride>
__forceinline__ __device__ void
insert_gradient_operator(double result[], const double recursion_factor) {
#pragma unroll
  for (int i = 0; i <= i_angular; i++) {
    double gradient, lower_order = 0;
#pragma unroll
    for (int j = 0; j <= j_angular; j++) {
      gradient =
          result[i * stride + j + 1] * recursion_factor - lower_order * j;
      lower_order = result[i * stride + j];
      result[i * stride + j] = gradient;
    }
  }
}

template <int i_angular, int j_angular, int stride>
__forceinline__ __device__ void
insert_gradient_operator_to_bra(double result[],
                                const double recursion_factor) {
#pragma unroll
  for (int j = 0; j <= j_angular; j++) {
    double gradient, lower_order = 0;
#pragma unroll
    for (int i = 0; i <= i_angular; i++) {
      gradient =
          result[(i + 1) * stride + j] * recursion_factor - lower_order * i;
      lower_order = result[i * stride + j];
      result[i * stride + j] = gradient;
    }
  }
}

template <int i_angular, int j_angular>
__forceinline__ __device__
void horizontal_recursion(double result[], const double shift_to_here) {
  constexpr int L = i_angular + j_angular, ncol = j_angular + 1;

  static_for<0, i_angular>([&](auto a_) {
    constexpr int a = decltype(a_)::value;
    static_for<0, L - a>([&](auto k_) {
      constexpr int b = L - a - 1 - decltype(k_)::value;
      result[(a+1)*ncol + b] = result[a*ncol + b+1] + shift_to_here * result[a*ncol + b];
    });
  });
}

template <int i_angular, int j_angular>
__forceinline__ __device__ void fill_with_recursion(
    double result[], const double prefactor, const double vrr_factor_prev,
    const double vrr_factor_second_prev, const double vrr_shift) {
  vertical_recursion<i_angular + j_angular>(result, prefactor, vrr_factor_prev,
                                            vrr_factor_second_prev);
  horizontal_recursion<i_angular, j_angular>(result, vrr_shift);
}
} // namespace rr
