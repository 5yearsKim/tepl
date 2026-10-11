#pragma once

#include <memory>
#include <string>
#include <vector>

#include "src/source.h"

namespace tepl::core::semantics {
// Scalar values are mathematical elements, distinct from metadata integers.
enum class Type {
  kScalar,
  kInteger,
  kBool,
  kVector,
  kTensor,
  kDomain,
  kAttrs,
  kOpaque
};
struct Expr;
using ExprPtr = std::shared_ptr<const Expr>;
struct Expr {
  std::string op;
  Type type;
  std::vector<ExprPtr> arguments;
  SourceOrigin origin;
};
// Resolved element semantics. Operation names and rewrite templates do not
// participate in lowering or verification.
struct Program {
  ExprPtr result;
  std::vector<std::size_t> used_fields;
};
std::string format(const Expr& expression);
}  // namespace tepl::core::semantics
