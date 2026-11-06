#pragma once

#include <map>
#include <optional>
#include <string>

#include "src/core/semantics/ir.h"
#include "src/shared/metadata/inference.h"
#include "src/shared/metadata/ordered.h"

namespace tepl::core::attr_synth::ordered {
using semantics::Definition;
using semantics::Plan;

// Recognized from checked definitions, never from operation/group/field names.
// This certifies the axis/value template. Dtype and arithmetic-policy
// obligations are still checked by the semantic verifier.
using Contract = axis::OrderedContract;

std::optional<Contract> recognize(const Definition&);

struct Admission {
  std::map<std::size_t, Contract> operations;
  inference::Program program;
  std::string reason;
  bool admitted() const { return reason.empty(); }
};

// Read-once components, matched capture sets, independent unrestricted RHS
// fields. This checks schema structure; concrete extent-domain checks are
// separate. Preserve original descriptor identities while selecting an expanded
// pair.
Plan componentPlan(const Plan&, const inference::Component&);
Admission admit(const Plan&, inference::Mode = inference::Mode::Synthesize, bool extensions = false);

}  // namespace tepl::core::attr_synth::ordered
