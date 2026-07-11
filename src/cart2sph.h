#ifndef CUINT_CART2SPH_H
#define CUINT_CART2SPH_H

#include <array>
#include <limits>
#include <numbers>

namespace c2s_helpers {

consteval double csqrt(double x) {
  if (x < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (x == 0.0 || x == std::numeric_limits<double>::infinity()) {
    return x;
  }
  
  double curr = x;
  double prev = x * 2.0;
  
  while (curr < prev) {
    prev = curr;
    curr = 0.5 * (curr + x / curr);
  }
  return curr;
}

// TODO use gamma function
consteval double cfactorial(int n) {
  double result = 1.0;
  for (int i = 2; i <= n; ++i) {
    result *= i;
  }
  return result;
}

consteval double ccomb(int n, int k) {
  if (k > n) return 0.0;
  if (k == 0 || k == n) return 1.0;
  if (k > n / 2) k = n - k;

  double result = 1.0;
  for (int i = 1; i <= k; ++i) {
    result *= (n - i + 1);
    result /= i;
  }
  return result;
}

consteval double cpow(double base, int exp) {
  if (exp == 0) return 1.0;
  if (exp < 0) return 1.0 / cpow(base, -exp);
  
  double result = 1.0;
  while (exp > 0) {
    if (exp % 2 == 1) result *= base;
    base *= base;
    exp /= 2;
  }
  return result;
}

template<typename T>
consteval int delta(const T& a, const T& b) {
  return (a == b) ? 1 : 0;
}

consteval double N(int l, int mu) {
  double fac = (2 - delta(mu, 0)) * (2 * l + 1) / (4 * std::numbers::pi);
  double result = csqrt(fac * cfactorial(l-mu) / cfactorial(l+mu));
  return result;
}

consteval double A(int l, int k, int mu) {
  double denom = cfactorial(k) * cfactorial(l-k) * cfactorial(l-mu-2*k);
  double result = cpow(-1, k) / cpow(2, l) * cfactorial(2*l-2*k) / denom;
  return result;
}

consteval double E(int r, int s, int mu) {
  if (s % 2 == 0 && r + s == mu) {
    return ccomb(mu, s) * cpow(-1, s / 2);
  } else {
    return 0.0;
  }
}

consteval double F(int r, int s, int mu) {
  if (s % 2 != 0 && r + s == mu) {
    return ccomb(mu, s) * cpow(-1, (s - 1) / 2);
  } else {
    return 0.0;
  }
}

consteval double H(int r, int s, int m) {
  if (r < 0 || s < 0) return 0.0;
  if (m == 0) {
    return delta(r, 0) * delta(s, 0);
  } else if (m > 0) {
    return E(r, s, m);
  } else {
    int mu = (m < 0) ? -m : m;
    return F(r, s, mu);
  }
}

} //namespace c2s_helpers

consteval double c2s(int l, int m, int lx, int ly, int lz) {
  using namespace c2s_helpers;

  if (l != lx + ly + lz) {
    throw "c2s: l must equal (lx+ly+lz)";
  }

  if (l == 1) {
#ifndef PYPZPX
    switch (m) {
      case -1:
        m = 1;
        break;
      case 0:
        m = -1;
        break;
      case 1:
        m = 0;
        break;
      default:
        throw "c2s: wrong m for l = 1";
        break;
    }
#endif
  }

  int mu = (m < 0) ? -m : m;
  if (mu > l) {
    throw "c2s: m > l is not allowed";
  }

  double result = 0.0;
  for (int k = 0; k <= (l - mu) / 2; ++k) {
    double tmp = 0.0;
    for (int u = 0; u <= k; ++u) {
      for (int v = 0; v <= k - u; ++v) {
        int w = k - u - v;
        int r = lx - 2 * u;
        int s = ly - 2 * v;
        tmp += cfactorial(k) / cfactorial(u) / cfactorial(v) / cfactorial(w) *
               H(r, s, m) * delta(lz, l-mu-2*k+2*w);
      }
    }
    result += A(l, k, mu) * tmp;
  }
  result *= N(l, mu);

  return result;
}

consteval int ncart(int l){
  return (l + 1) * (l + 2) / 2;
}

template <int l>
consteval std::array<std::array<double, ncart(l)>, 2*l+1> c2s_matrix() {
  std::array<std::array<double, ncart(l)>, 2*l+1> coeff{};
  for (int m = -l; m <= l; ++m) {
    int c = 0;
    for (int lx = l; lx >= 0; --lx)
      for (int ly = l - lx; ly >= 0; --ly)
        coeff[m + l][c++] = c2s(l, m, lx, ly, l - lx - ly);
  }
  return coeff;
}

template <int l>
consteval std::array<std::array<int,3>, ncart(l)> cart_ls() {
  std::array<std::array<int,3>, ncart(l)> t{};
  int c = 0;
  for (int lx = l; lx >= 0; --lx)
    for (int ly = l - lx; ly >= 0; --ly)
      t[c++] = {lx, ly, l - lx - ly};
  return t;
}

#endif
