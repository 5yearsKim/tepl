#pragma once

#include <cstddef>
#include <string>

namespace tepl::core::shape {

// One integer domain covers u64 dimensions and signed i64 attributes. Generated
// evaluators use checked signed 128-bit arithmetic and validate final
// dimensions against u64. Lists are homogeneous and can nest to any depth.
struct Type {
  enum class Kind { kInteger, kBoolean } kind = Kind::kInteger;
  std::size_t list_depth = 0;
  bool operator==(const Type&) const = default;
};

std::string typeName(Type type);
bool validIntegerLiteral(const std::string& spelling);

}  // namespace tepl::core::shape
