#pragma once

#include "utils.h"
#include "cart2sph.h"

namespace ovlp {

template <int i_angular, int j_angular, int stride>
__forceinline__ __device__ void
write_integral(double *output, const double x_pairs[], const double y_pairs[],
               const double z_pairs[], const int n_functions) {

  constexpr auto Ci = c2s_matrix<i_angular>();
  constexpr auto Bi = cart_ls<i_angular>();
  constexpr auto Cj = c2s_matrix<j_angular>();
  constexpr auto Bj = cart_ls<j_angular>();

  static_for<0, 2*i_angular+1>([&](auto ioff_) {
    constexpr int ioff = decltype(ioff_)::value;
    static_for<0, 2*j_angular+1>([&](auto joff_) {
      constexpr int joff = decltype(joff_)::value;
      double expression = 0.0;

      static_for<0, ncart(i_angular)>([&](auto i_){
        constexpr int i = decltype(i_)::value;
        constexpr double ci = Ci[ioff][i];
        constexpr double abs_ci = (ci < 0.0) ? -ci : ci;
        if constexpr (abs_ci > 1e-15) {
          constexpr int xi=Bi[i][0], yi=Bi[i][1], zi=Bi[i][2];
          static_for<0, ncart(j_angular)>([&](auto j_){
            constexpr int j = decltype(j_)::value;
            constexpr double cj = Cj[joff][j];
            constexpr double abs_cj = (cj < 0.0) ? -cj : cj;
            if constexpr (abs_cj > 1e-15) {
              constexpr double cij = ci * cj;
              constexpr int xj=Bj[j][0], yj=Bj[j][1], zj=Bj[j][2];
              expression += cij * x_pairs[xi * stride + xj] *
                            y_pairs[yi * stride + yj] * z_pairs[zi * stride + zj];
            }
          });
        }
      });

      atomicAdd(output + ioff * n_functions + joff, expression);
    });
  });
}

} // namespace ovlp
