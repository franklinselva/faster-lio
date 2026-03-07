#ifndef FASTER_LIO_COMPAT_H
#define FASTER_LIO_COMPAT_H

#include <algorithm>

// C++17 parallel execution policy support.
// CMake sets FASTER_LIO_HAS_STD_EXECUTION when <execution> + TBB backend works.
// On Linux/GCC with TBB: parallel std::for_each via TBB thread pool.
// On macOS/Apple Clang: sequential fallback (no <execution> support).
#ifdef FASTER_LIO_HAS_STD_EXECUTION

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

#else

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

#endif

#endif  // FASTER_LIO_COMPAT_H
