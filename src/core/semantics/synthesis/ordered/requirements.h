#pragma once

#include "src/core/semantics/symbolic/normalize.h"
#include "src/core/semantics/synthesis/ordered/contract.h"

namespace tepl::core::semantics::ordered {

struct NodeRequirements {
  S live;
  S batch, contract, left_free, right_free;
  S required_order;  // Only a subsequence requirement, never a chosen layout.
};

struct Requirements {
  std::map<std::size_t, NodeRequirements> nodes;
  std::vector<S> order_constraints;
};

// Derives roles and propagates order requirements without assigning attributes.
// Predicates are obligations, not new premises or automatically inserted
// guards.
Requirements deriveRequirements(Context&, const Admission&,
                                const Correspondence&, const S& output);

}  // namespace tepl::core::semantics::ordered
