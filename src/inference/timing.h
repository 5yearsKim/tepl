#pragma once
#include <chrono>

namespace tepl::inference {
// Optional execution profiling. Source replay and surrounding setup/teardown
// are outside these intervals; allocations within q/f stay in those phases.
// Timings are accumulated across component blocks.
struct ExecutionTiming {
  double feasibility_us = 0;
  double construction_us = 0;
};
namespace detail {
class PhaseTimer {
 public:
  explicit PhaseTimer(double* result) : result_(result) {
    if (result_) start_ = Clock::now();
  }
  ~PhaseTimer() {
    if (result_)
      *result_ +=
          std::chrono::duration<double, std::micro>(Clock::now() - start_)
              .count();
  }

 private:
  using Clock = std::chrono::steady_clock;
  double* result_;
  Clock::time_point start_;
};
}  // namespace detail
}  // namespace tepl::inference
