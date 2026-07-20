#pragma once

#include <array>

#include "utils.h"

// McMurchie-Davidson building blocks: the Hermite Gaussian index algebra,
// the pair expansion coefficients E^p_ab, the geometry-free single-center
// coefficients E^q_c, and the Hermite Coulomb tower [r]^(0).
namespace md {

// number of Hermite triples (t, u, v) with t + u + v <= l
__host__ __device__ constexpr int nherm(const int l) {
  return (l + 1) * (l + 2) * (l + 3) / 6;
}

// linear index of the triple (t, u, v): shells |r| = t + u + v ascending,
// within a shell t descending, then u descending
__host__ __device__ constexpr int hindex(const int t, const int u, const int v) {
  const int s = t + u + v;
  return s * (s + 1) * (s + 2) / 6 + (s - t) * (s - t + 1) / 2 + v;
}

struct Triple {
  int t, u, v;
};

// inverse of hindex
__host__ __device__ constexpr Triple herm_triple(const int index) {
  int s = 0;
  while (nherm(s) <= index) {
    s++;
  }
  const int offset = index - s * (s + 1) * (s + 2) / 6;
  int k = 0;
  while ((k + 1) * (k + 2) / 2 <= offset) {
    k++;
  }
  const int v = offset - k * (k + 1) / 2;
  return {s - k, k - v, v};
}

consteval int leading_axis(const Triple r) {
  return r.t > 0 ? 0 : (r.u > 0 ? 1 : 2);
}

consteval Triple lowered(const Triple r, const int axis) {
  return axis == 0   ? Triple{r.t - 1, r.u, r.v}
         : axis == 1 ? Triple{r.t, r.u - 1, r.v}
                     : Triple{r.t, r.u, r.v - 1};
}

consteval int component(const Triple r, const int axis) {
  return axis == 0 ? r.t : (axis == 1 ? r.u : r.v);
}

// Precomputed index algebra for the table-driven two-stage kernels: entry i
// caches herm_triple(i), the leading axis, and the hindex of the once- and
// twice-lowered triples, replacing per-element herm_triple_rt while-loops.
struct HermEntry {
  short low1, low2;   // hindex of the once/twice-lowered triple
  signed char axis;   // leading axis of the triple
  signed char comp;   // component(once-lowered, axis); 0 means no 2nd term
  signed char t, u, v;
};

template <int N>
consteval std::array<HermEntry, N> make_herm_entries() {
  std::array<HermEntry, N> entries{};
  for (int i = 1; i < N; i++) {
    const Triple p = herm_triple(i);
    const int axis = leading_axis(p);
    const Triple p1 = lowered(p, axis);
    const int comp = component(p1, axis);
    const Triple p2 = comp > 0 ? lowered(p1, axis) : p1;
    entries[i] = {(short)hindex(p1.t, p1.u, p1.v),
                  (short)hindex(p2.t, p2.u, p2.v),
                  (signed char)axis,
                  (signed char)comp,
                  (signed char)p.t,
                  (signed char)p.u,
                  (signed char)p.v};
  }
  return entries;
}

// Hermite Coulomb tower: r[hindex(p)] = [p]^(0) for all |p| <= L, built in
// place from the seeded Boys values f[m] = prefactor * (-2 rho)^m F_m(T).
// Pseudo code:
// Loop m from L to 0:
//  Loop s from L-m to 1:
//    r[s] = [s]^(m)
//  r[0] = f[m]
// Schematic diagram:
// m=L  : [0]
// m=L-1: [0], [1]
// m=L-2: [0], [1], [2]
//  ⋮      ⋮    ⋮    ⋮
// m=0  : [0], [1], [2], …, [L]
template <int L>
__forceinline__ __device__ void hermite_tower(double (&r)[nherm(L)],
                                              const double (&f)[L + 1],
                                              const double w_x,
                                              const double w_y,
                                              const double w_z) {
  r[0] = f[L]; // [0]^(L)

  static_for<0, L>([&]<int step>() {
    constexpr int m = L - 1 - step;

    static_for<0, L - m>([&]<int down>() {
      constexpr int s = L - m - down;

      static_for<nherm(s - 1), nherm(s)>([&]<int i>() {
        constexpr Triple p = herm_triple(i);
        constexpr int axis = leading_axis(p);
        constexpr Triple p1 = lowered(p, axis);

        const double w = axis == 0 ? w_x : (axis == 1 ? w_y : w_z);
        double value = w * r[hindex(p1.t, p1.u, p1.v)];
        if constexpr (component(p1, axis) > 0) {
          constexpr Triple p2 = lowered(p1, axis);
          value += component(p1, axis) * r[hindex(p2.t, p2.u, p2.v)];
        }
        r[i] = value;
      });
    });

    r[0] = f[m];
  });
}

// Number of live pair coefficients E^p_ab: only p <= a + b is used, so the
// dense box (LA+1)(LB+1)(LA+LB+1) is packed to its triangular subset.
template <int LA, int LB>
consteval int nepair() {
  return (LA + 1) * (LB + 1) * (LA + LB + 2) / 2;
}

template <int LA, int LB>
__host__ __device__ constexpr int eindex(const int a, const int b, const int p) {
  int offset = 0; // reserve a + b + 1 slots per (a, b), in a-major order
  for (int aa = 0; aa < a; aa++)
    for (int bb = 0; bb <= LB; bb++)
      offset += aa + bb + 1;
  for (int bb = 0; bb < b; bb++)
    offset += a + bb + 1;
  return offset + p;
}

// Pair expansion coefficients along one axis, e[eindex(a, b, p)] = E^p_ab,
// seeded with E^0_00 = 1 (the Gaussian prefactor exp(-mu X^2) is folded into
// the Boys values instead). i_to_p = P - A and j_to_p = P - B on this axis;
// half_over_zeta = 1 / (2 zeta_p). Entries with p > a + b are never written
// nor read.
template <int LA, int LB>
__forceinline__ __device__ void e_pair(
    double (&e)[nepair<LA, LB>()], const double i_to_p,
    const double j_to_p, const double half_over_zeta) {
  e[eindex<LA, LB>(0, 0, 0)] = 1.0;

  // raise a at b = 0
  static_for<0, LA>([&]<int a>() {
    static_for<0, a + 2>([&]<int p>() {
      double value = 0.0;
      if constexpr (p <= a) {
        value += i_to_p * e[eindex<LA, LB>(a, 0, p)];
      }
      if constexpr (p > 0) {
        value += half_over_zeta * e[eindex<LA, LB>(a, 0, p - 1)];
      }
      if constexpr (p + 1 <= a) {
        value += (p + 1) * e[eindex<LA, LB>(a, 0, p + 1)];
      }
      e[eindex<LA, LB>(a + 1, 0, p)] = value;
    });
  });

  // raise b for all a
  static_for<0, LB>([&]<int b>() {
    static_for<0, LA + 1>([&]<int a>() {
      static_for<0, a + b + 2>([&]<int p>() {
        double value = 0.0;
        if constexpr (p <= a + b) {
          value += j_to_p * e[eindex<LA, LB>(a, b, p)];
        }
        if constexpr (p > 0) {
          value += half_over_zeta * e[eindex<LA, LB>(a, b, p - 1)];
        }
        if constexpr (p + 1 <= a + b) {
          value += (p + 1) * e[eindex<LA, LB>(a, b, p + 1)];
        }
        e[eindex<LA, LB>(a, b + 1, p)] = value;
      });
    });
  });
}

consteval double factorial(const int n) {
  double result = 1.0;
  for (int i = 2; i <= n; i++) {
    result *= i;
  }
  return result;
}

// Single-center (geometry-free) expansion coefficients: along one axis,
// E^q_c = aux_coef(c, q) * (2 zeta_c)^{-(c + q) / 2}, nonzero only for
// q = c, c - 2, ..., following from the closed form with t = (c - q) / 2.
consteval double aux_coef(const int c, const int q) {
  const int t = (c - q) / 2;
  double two_to_t = 1.0;
  for (int i = 0; i < t; i++) {
    two_to_t *= 2.0;
  }
  return factorial(c) / (factorial(q) * factorial(t) * two_to_t);
}

}  // namespace md
