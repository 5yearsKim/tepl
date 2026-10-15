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
struct AttributeField {
  std::string name;
  std::string type;
  std::size_t list_depth = 0;
  bool optional = false;
  bool empty_default = false;
  SourceOrigin origin;
};

struct AttributeSchema {
  AttributeSchemaId id;
  std::string dialect;
  std::string name;
  std::vector<AttributeField> fields;
  SourceOrigin origin;
};

struct Operand {
  std::string name;
  TypeId type;
  bool variadic = false;
  SourceOrigin origin;
};

struct OperationSignature {
  OpId id;
  std::string dialect;
  std::string name;
  std::optional<std::string> alias;
  std::vector<Operand> operands;
  TypeId result;
  std::optional<AttributeSchemaId> attributes;
  SourceOrigin origin;
  std::optional<metadata::Program> shape;
  std::optional<metadata::Program> dtype;
};

struct Dialect {
  std::string name;
  SourceOrigin origin;
};

}  // namespace tepl::core
