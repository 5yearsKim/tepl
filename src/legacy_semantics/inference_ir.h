#pragma once
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/core/ids.h"
#include "src/legacy_semantics/plan.h"

namespace tepl::core::inference {
// A small, pure attribute-building language. It never examines tensor values.
struct Value;
using ValuePtr = std::shared_ptr<const Value>;
struct Value {
  enum class Kind { kLiteral, kField, kShape, kDType, kSemanticWitness } kind;
  std::string text;
  std::vector<ValuePtr> arguments;
  DescriptorId descriptor{};
  AttributeSchemaId schema{};
  std::size_t field = 0;
  CaptureId capture{};
  bool string_literal = false;
};
struct Assignment {
  DescriptorId descriptor;
  AttributeSchemaId schema;
  std::vector<ValuePtr> fields;
};
struct Program {
  std::vector<Assignment> assignments;
  // Selection does not claim uniqueness. Guards can restrict applicability.
  std::string selection = "canonical";
  std::string multiplicity = "not_checked";
  std::string coverage = "guarded";
  std::vector<std::string> proof;
  std::optional<SemanticPlan> semantic;
  std::vector<ValuePtr> pinned;
};
}  // namespace tepl::core::inference
