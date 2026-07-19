#pragma once

#include "boys_table.h"
#include "utils.h"

// Boys function F_m(x) = int_0^1 t^{2m} exp(-x t^2) dt for m = 0..M.
//
// Below the switch point, F_M is read from a degree-7 Chebyshev table
// (0.2-wide intervals, verified to < 3e-15 relative error) and the lower
// orders follow from the downward recursion; the reciprocals are
// compile-time constants, so the whole path is division-free straight-line
// code. Above the switch point the asymptotic value seeds F_0 and the
// upward recursion is used. Both paths are short, so a warp that straddles
// the switch point pays little.
template <int M>
__forceinline__ __device__ void boys_function(const double x,
                                              double (&f)[M + 1]) {
  static_assert(M <= BOYS_TABLE_M_MAX, "boys_table.h does not cover this M");

  if (x < BOYS_SWITCH_X) {
    const int interval = (int)(x * BOYS_INTERVAL_INV_H);
    const double *c = boys_table + (M * BOYS_N_INTERVALS + interval) * 8;
    const double u =
        (x - (interval + 0.5) * BOYS_INTERVAL_H) * (2.0 * BOYS_INTERVAL_INV_H);

    double value = c[7];
    static_for<0, 7>([&]<int k>() {
      value = fma(value, u, c[6 - k]);
    });
    f[M] = value;

    if constexpr (M > 0) {
      const double exp_minus_x = exp(-x);
      static_for<0, M>([&]<int k>() {
        constexpr int m = M - 1 - k;
        f[m] = (2.0 * x * f[m + 1] + exp_minus_x) * (1.0 / (2 * m + 1));
      });
    }
  } else {
    f[0] = 0.88622692545275801365 * rsqrt(x);
    const double half_over_x = 0.5 / x;
    static_for<0, M>([&]<int m>() {
      f[m + 1] = ((2 * m + 1) * f[m]) * half_over_x; // -exp(-x) dropped for large x
    });
  }
}
