#pragma once
#include <functional>

#include "src/executable/program.h"
namespace tepl::rewrite {
struct Options {
  std::string rule;
  std::size_t max_steps = 1000, max_nodes = 1000000,
              max_expression_steps = 1000000, max_list_items = 1000000;
  bool trace = false;
  std::function<bool()> cancelled;
};
enum class StopReason { FixedPoint, StepLimit, ResourceLimit, Cancelled };
std::string_view stopReasonName(StopReason);
struct Step {
  std::string rule;
  std::size_t node;
};
struct Result {
  core::ConcreteGraph graph;
  StopReason stop_reason = StopReason::FixedPoint;
  std::size_t rewrite_count = 0;
  std::vector<Step> trace;
  std::string detail;
};
// Copies the input; every replacement is validated before commit.
Result run(const executable::ExecutableProgram&, const core::ConcreteGraph&,
           const Options& = {});
}  // namespace tepl::rewrite
