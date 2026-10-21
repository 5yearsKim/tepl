#pragma once

#include <map>
#include <optional>
#include <string>

#include "src/core/semantics/ir.h"
#include "src/shared/metadata/inference.h"
#include "src/shared/metadata/ordered.h"

namespace tepl::core::semantics::ordered {

// Recognized from checked definitions, never from operation/group/field names.
// This certifies the axis/value template. Dtype and arithmetic-policy
// obligations are still checked by the semantic verifier.
using Contract = axis::OrderedContract;

std::optional<Contract> recognize(const Definition&);

struct Admission {
  std::map<std::size_t, Contract> operations;
  std::string reason;
  bool admitted() const { return reason.empty(); }
};

// Read-once trees, equal named captures, independent unrestricted RHS fields.
// This checks schema structure; concrete extent-domain checks are separate.
Admission admit(const Plan&, inference::Mode = inference::Mode::Synthesize);

}  // namespace tepl::core::semantics::ordered
