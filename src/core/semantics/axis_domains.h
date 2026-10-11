#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "src/core/inference/semantic_ir.h"

namespace tepl::core::semantics {
// Domains describe active entries only. Optional padded entries are
// deliberately unconstrained until their position is known to be active.
struct AxisDomain {
  std::vector<std::uint64_t> lengths;
  std::vector<std::vector<std::uint64_t>> entries;
};
struct AxisPropagation {
  bool available = false;
  bool inconsistent = false;
  std::string reason;
  std::map<std::size_t, AxisDomain> fields;
  std::size_t original_entries = 0, forced_entries = 0;
  std::size_t residual_entries = 0, instructions = 0;
};
// Interpret the compiled primitive DAG as finite-domain relations, without Z3.
// Restrictions belong to the sufficient semantic matcher, not to a completeness
// claim about tensor equality. Failed/unsupported preprocessing must be
// discarded.
AxisPropagation propagateAxes(
    const inference::SemanticPlan&,
    const std::vector<std::vector<std::uint64_t>>& shapes,
    const std::vector<std::vector<std::uint64_t>>& known,
    const std::vector<std::vector<std::uint64_t>>& pinned,
    const std::function<void()>& tick);
}  // namespace tepl::core::semantics
