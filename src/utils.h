#ifndef CUINT_UTILS_H
#define CUINT_UTILS_H

#include <utility>

template <int Begin, int End, typename F>
__host__ __device__ __forceinline__ constexpr void static_for(F&& f) {
  if constexpr (Begin < End) {
    [&]<int... Is>(std::integer_sequence<int, Is...>) {
      (f.template operator()<Begin + Is>(), ...);
    }
    (std::make_integer_sequence<int, End - Begin>{});
  }
}

consteval int pow3(const int n) {
  int result = 1;
  for (int i = 0; i < n; i++) {
    result *= 3;
  }
  return result;
}

// Derivative components enumerate the Cartesian product of (x, y, z) over
// n_slots derivative slots, with slot 0 varying slowest. Returns how many of
// the slots in [slot_begin, slot_end) of component c point along `axis`
// (0 = x, 1 = y, 2 = z).
consteval int deriv_count(const int c, const int n_slots, const int slot_begin,
                          const int slot_end, const int axis) {
  int count = 0;
  for (int slot = slot_begin; slot < slot_end; slot++) {
    if (c / pow3(n_slots - 1 - slot) % 3 == axis) {
      count++;
    }
  }
  return count;
}

template <int Begin, int End, typename F>
void dispatch_range(const int i, F&& f) {
  if constexpr (Begin < End) {
    [&]<int... Is>(std::integer_sequence<int, Is...>) {
      ((Begin + Is == i ? (f.template operator()<Begin + Is>(), true)
                        : false) ||
       ...);
    }
    (std::make_integer_sequence<int, End - Begin>{});
  }
}

#endif
