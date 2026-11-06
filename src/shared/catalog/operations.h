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
enum class AttributeRole { General, Axis, Extent, Coordinate };

struct AttributeField {
  std::string name;
  std::string type;
  std::size_t list_depth = 0;
  bool optional = false;
  bool empty_default = false;
  SourceOrigin origin;
  // Compile-time role; executable descriptors retain their integer ABI.
  AttributeRole role = AttributeRole::General;

  Type valueType() const {
    auto value =
        resolveType(type).value_or(Type{TypeKind::kOpaque, {}, 0, false, type});
    for (std::size_t d = 0; d < list_depth; ++d) value = listType(value);
    value.optional = optional;
    return value;
  }
  enum class Default { EmptyList, None };
  std::optional<Default> defaultKind() const {
    if (empty_default) return Default::EmptyList;
    if (optional) return Default::None;
    return {};
  }
  bool allowsAuto() const {
    return role == AttributeRole::Axis && !optional && list_depth <= 1;
  }
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
