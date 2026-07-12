#ifndef CUINT_UTILS_H
#define CUINT_UTILS_H

#include <utility>

template <int Begin, int End, typename F>
__host__ __device__ __forceinline__ constexpr void static_for(F&& f) {
  if constexpr (Begin < End) {
    [&]<int... Is>(std::integer_sequence<int, Is...>) {
      (f.template operator()<Begin + Is>(), ...);
    }(std::make_integer_sequence<int, End - Begin>{});
  }
}

template <int Begin, int End, typename F>
void dispatch_range(const int i, F&& f) {
  if constexpr (Begin < End) {
    [&]<int... Is>(std::integer_sequence<int, Is...>) {
      ((Begin + Is == i ? (f.template operator()<Begin + Is>(), true) : false) || ...);
    }(std::make_integer_sequence<int, End - Begin>{});
  }
}

#endif
