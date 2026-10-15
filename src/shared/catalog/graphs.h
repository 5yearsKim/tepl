#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "src/shared/catalog/ids.h"
#include "src/shared/metadata/axes.h"
#include "src/shared/metadata/ir.h"
#include "src/shared/source/source.h"
#include "src/shared/types/operators.h"
#include "src/shared/types/types.h"

namespace tepl::core {
// Checked concrete graphs are topologically ordered DAGs, not rewrite patterns.
struct AttributeValue {
  enum class Kind { kInteger, kBool, kString, kEnum, kList, kNone } kind;
  std::string text;
  std::vector<AttributeValue> elements;
};
struct ConcreteNode {
  enum class Kind { kInput, kLiteral, kOperation } kind;
  SourceOrigin origin;
  std::string text;
  std::optional<DType> dtype;
  std::optional<std::vector<std::uint64_t>> shape;
  std::optional<OpId> operation;
  std::vector<std::size_t> operands;
  // Schema field order, including declared defaults.
  std::vector<AttributeValue> attributes;
};
struct ConcreteGraph {
  GraphId id;
  std::string name, source_name;
  SourceOrigin origin;
  std::vector<ConcreteNode> nodes;
  std::vector<std::pair<std::string, std::size_t>> bindings;
  std::size_t root = 0;
};

}  // namespace tepl::core
