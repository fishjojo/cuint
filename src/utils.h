#include <utility>
#include <type_traits>

template <int Begin, int End, class F>
__host__ __device__ __forceinline__
constexpr void static_for(F&& f) {
  if constexpr (Begin < End) {
    f(std::integral_constant<int, Begin>{});
    static_for<Begin + 1, End>(std::forward<F>(f));
  }
}

