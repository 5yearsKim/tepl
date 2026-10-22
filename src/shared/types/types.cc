#include "src/shared/types/types.h"

#include <utility>

namespace tepl::core {
namespace {
constexpr std::pair<std::string_view, DType> kDTypes[] = {
    {"bool", DType::kBool}, {"i8", DType::kI8},     {"i16", DType::kI16},
    {"i32", DType::kI32},   {"i64", DType::kI64},   {"u8", DType::kU8},
    {"u16", DType::kU16},   {"u32", DType::kU32},   {"u64", DType::kU64},
    {"f16", DType::kF16},   {"bf16", DType::kBF16}, {"f32", DType::kF32},
    {"f64", DType::kF64},
};
}  // namespace

std::optional<DType> resolveDType(std::string_view name) {
  for (auto [spelling, dtype] : kDTypes)
    if (name == spelling) return dtype;
  return std::nullopt;
}

std::string_view dtypeName(DType dtype) {
  for (auto [spelling, value] : kDTypes)
    if (value == dtype) return spelling;
  return "<invalid dtype>";
}

Type logicalAxisSetType() {
  Type t{TypeKind::kOpaque};
  t.opaque_name = "LogicalAxisSet";
  return t;
}
Type logicalAxisType() {
  Type t{TypeKind::kOpaque};
  t.opaque_name = "LogicalAxis";
  return t;
}

Type listType(Type element) {
  if (element.kind == TypeKind::kIndexList) {
    element.kind = TypeKind::kIndex;
    ++element.list_depth;
  }
  ++element.list_depth;
  if (element.kind == TypeKind::kIndex && element.list_depth == 1 &&
      !element.optional) {
    element.kind = TypeKind::kIndexList;
    element.list_depth = 0;
  }
  return element;
}
std::optional<Type> elementType(Type list) {
  if (list.optional) return {};
  if (list.kind == TypeKind::kIndexList) return Type{TypeKind::kIndex};
  if (!list.list_depth) return {};
  --list.list_depth;
  if (list.kind == TypeKind::kIndex && list.list_depth == 1) {
    list.kind = TypeKind::kIndexList;
    list.list_depth = 0;
  }
  return list;
}
std::string typeName(const Type& type) {
  if (type.list_depth || type.optional) {
    auto scalar = type;
    scalar.list_depth = 0;
    scalar.optional = false;
    auto result = typeName(scalar);
    for (std::size_t i = 0; i < type.list_depth; ++i) result += "[]";
    return result + (type.optional ? "?" : "");
  }
  switch (type.kind) {
    case TypeKind::kString:
      return "String";
    case TypeKind::kPrecision:
      return "Precision";
    case TypeKind::kOpaque:
      return type.opaque_name;
    case TypeKind::kTensor:
      return "Tensor";
    case TypeKind::kIndex:
      return "Index";
    case TypeKind::kIndexList:
      return "IndexList";
    case TypeKind::kBool:
      return "Bool";
    case TypeKind::kI64:
      return "I64";
    case TypeKind::kF64:
      return "F64";
    case TypeKind::kDType:
      return "DType";
    case TypeKind::kDescriptor:
      return type.schema
                 ? "Descriptor(#" + std::to_string(type.schema->value) + ")"
                 : "Descriptor";
  }
  return "<invalid type>";
}

std::optional<Type> resolveType(std::string_view name) {
  static constexpr std::pair<std::string_view, TypeKind> names[] = {
      {"dtype", TypeKind::kDType},   {"tensor", TypeKind::kTensor},
      {"index", TypeKind::kIndex},   {"index_list", TypeKind::kIndexList},
      {"string", TypeKind::kString}, {"precision", TypeKind::kPrecision},
      {"bool", TypeKind::kBool},     {"i64", TypeKind::kI64},
      {"f64", TypeKind::kF64},       {"attrs", TypeKind::kDescriptor},
  };
  for (const auto& [spelling, kind] : names)
    if (name == spelling) return Type{kind};
  return std::nullopt;
}

bool isInteger(TypeKind kind) {
  return kind == TypeKind::kIndex || kind == TypeKind::kI64;
}

bool isNumeric(TypeKind kind) {
  return isInteger(kind) || kind == TypeKind::kF64;
}

}  // namespace tepl::core
