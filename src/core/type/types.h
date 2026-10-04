#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/ids.h"

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

// Host values are distinct from graph tensors and their element dtypes.
// Index and I64 have unsigned/signed 64-bit literal ranges in this IR.
enum class TypeKind {
  kTensor,
  kIndex,
  kIndexList,
  kBool,
  kI64,
  kF64,
  kDescriptor
};

struct Type {
  TypeKind kind = TypeKind::kTensor;
  // Missing schema denotes an opaque host descriptor. Graph descriptor uses
  // always resolve to a concrete schema.
  std::optional<AttributeSchemaId> schema;
  bool operator==(const Type&) const = default;
};

struct Signature {
  std::vector<TypeId> arguments;
  TypeId result;
};

std::string typeName(const Type& type);
std::optional<Type> resolveType(std::string_view name);
bool isInteger(TypeKind kind);
bool isNumeric(TypeKind kind);

}  // namespace tepl::core
