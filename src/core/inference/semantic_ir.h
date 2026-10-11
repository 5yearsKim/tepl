#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/core/ids.h"
namespace tepl::core::inference {
// Shared bytecode for metadata constraints and tensor element semantics.
struct SemanticExpr {
  std::string op;
  std::vector<std::size_t> args;
};
struct SemanticField {
  int known = -1;
  std::size_t output = 0;
  // An explicit is_valid_axis_list(field, bound) supplies a finite domain.
  // This is an expression index, not an operation-specific axis role.
  std::size_t bound = 0;
};
struct SemanticNode {
  int capture = -1;
  std::vector<std::size_t> inputs;
  std::vector<SemanticExpr> expressions;
  std::size_t root = 0, shape = 0;
  std::vector<std::size_t> assertions;
  std::vector<SemanticField> fields;
};
struct KnownField {
  DescriptorId descriptor;
  AttributeSchemaId schema;
  std::size_t field;
};
struct ConstraintRead {
  std::size_t capture = 0;
  std::size_t indices = 0;
};
struct ConstraintTerm {
  std::int64_t coefficient = 1;
  std::vector<std::size_t> reductions;
  std::vector<ConstraintRead> reads;
  // Subset of reductions whose coordinates occur in no access expression.
  // Match-time constraints may eliminate them only for one-point domains.
  std::vector<std::size_t> unused_reductions;
};
// A compiled sufficient condition for semantic equality. Vector lengths and
// domains remain symbolic until a match. Failed matching is never an
// impossibility proof; the original semantics remain available for
// verification.
struct ConstraintPlan {
  bool available = false;
  std::string reason;
  // @shape:N / @attr:N:F reference metadata. @output / @bound:N reference
  // coordinate vectors; binder identities are globally unique in this plan.
  std::vector<SemanticExpr> expressions;
  std::vector<std::size_t> domains;
  std::vector<ConstraintTerm> lhs, rhs;
};
struct SemanticPlan {
  std::vector<CaptureId> captures;
  std::vector<SemanticNode> nodes;
  std::vector<KnownField> known;
  std::size_t source_root = 0, rhs_root = 0, source_end = 0;
  bool algebraic = false;
  unsigned timeout_ms = 1000;
  ConstraintPlan constraints;
};
}  // namespace tepl::core::inference
