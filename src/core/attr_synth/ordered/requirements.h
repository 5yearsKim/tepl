#pragma once

#include "src/core/attr_synth/ordered/contract.h"
#include "src/core/semantics/symbolic/normalize.h"

namespace tepl::core::attr_synth::ordered {
using semantics::Context;
using semantics::Correspondence;
using semantics::S;

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

}  // namespace tepl::core::attr_synth::ordered
