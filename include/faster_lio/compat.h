#ifndef FASTER_LIO_COMPAT_H
#define FASTER_LIO_COMPAT_H

#include <algorithm>

// Apple Clang does not support C++17 parallel execution policies.
// On macOS, fall back to sequential std::for_each.
#if defined(__APPLE__) || !defined(__cpp_lib_execution)

namespace faster_lio::compat {
struct sequenced_policy {};
struct parallel_unsequenced_policy {};
struct unsequenced_policy {};

inline constexpr sequenced_policy seq{};
inline constexpr parallel_unsequenced_policy par_unseq{};
inline constexpr unsequenced_policy unseq{};

template <typename Policy, typename It, typename F>
void for_each(Policy&&, It first, It last, F f) {
    std::for_each(first, last, f);
}
}  // namespace faster_lio::compat

#else

#include <execution>
namespace faster_lio::compat {
using std::execution::par_unseq;
using std::execution::seq;
inline constexpr auto unseq = std::execution::unseq;

template <typename Policy, typename It, typename F>
void for_each(Policy&& policy, It first, It last, F f) {
    std::for_each(std::forward<Policy>(policy), first, last, f);
}
}  // namespace faster_lio::compat

#endif

#endif  // FASTER_LIO_COMPAT_H
