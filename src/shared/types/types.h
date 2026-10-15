#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/shared/catalog/ids.h"

namespace tepl::core {

enum class DType {
  kBool,
  kI8,
  kI16,
  kI32,
  kI64,
  kU8,
  kU16,
  kU32,
  kU64,
  kF16,
  kBF16,
  kF32,
  kF64,
};

std::optional<DType> resolveDType(std::string_view name);
std::string_view dtypeName(DType dtype);

// Rule values are distinct from graph tensors and their element dtypes.
// Index and I64 have unsigned/signed 64-bit literal ranges in this IR.
enum class TypeKind {
  kTensor,
  kIndex,
  kIndexList,
  kBool,
  kI64,
  kF64,
  kDescriptor,
  kDType,
  kString,
  kPrecision,
  kOpaque
};

struct Type {
  TypeKind kind = TypeKind::kTensor;
  // Descriptor values resolve to a concrete attribute schema.
  std::optional<AttributeSchemaId> schema;
  std::size_t list_depth = 0;
  bool optional = false;
  std::string opaque_name;
  bool operator==(const Type&) const = default;
};

Type listType(Type element);
std::optional<Type> elementType(Type list);
std::string typeName(const Type& type);
std::optional<Type> resolveType(std::string_view name);
bool isInteger(TypeKind kind);
bool isNumeric(TypeKind kind);

}  // namespace tepl::core
