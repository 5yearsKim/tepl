#pragma once
#include <memory>
#include <string>
#include <vector>

#include "src/core/ids.h"

namespace tepl::core {
struct AttributePin;
using AttributePinPtr = std::shared_ptr<const AttributePin>;
struct AttributePin {
  enum class Kind { kLiteral, kField } kind;
  std::string text;
  std::vector<AttributePinPtr> arguments;
  DescriptorId descriptor{};
  AttributeSchemaId schema{};
  std::size_t field = 0;
  bool string_literal = false;
};
struct AttributeHole {
  DescriptorId descriptor;
  AttributeSchemaId schema;
  std::vector<AttributePinPtr> overrides;
};
}  // namespace tepl::core
