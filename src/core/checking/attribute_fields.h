#pragma once

#include "src/core/checking/rule_ir.h"
#include "src/shared/catalog/operations.h"

namespace tepl::core {
// Omitted and explicit constructors use the same typed schema defaults.
// A null result denotes a required field, which the caller must supply.
template <class InternType>
TypedExprPtr attributeDefault(const AttributeField& field,
                              const SourceOrigin& origin, InternType typeId) {
  const auto kind = field.defaultKind();
  if (!kind) return {};
  const auto type = field.valueType();
  if (*kind == AttributeField::Default::None)
    return std::make_shared<TypedExpr>(
        TypedExpr{origin, typeId(type), NoneConstant{}});
  auto present = type;
  present.optional = false;
  auto empty = std::make_shared<TypedExpr>(
      TypedExpr{origin, typeId(present), ListExpr{}});
  if (!type.optional) return empty;
  return std::make_shared<TypedExpr>(
      TypedExpr{origin, typeId(type), OptionalValue{empty}});
}
}  // namespace tepl::core
