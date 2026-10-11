#pragma once
#include <cstddef>
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
struct SemanticPlan {
  std::vector<CaptureId> captures;
  std::vector<SemanticNode> nodes;
  std::vector<KnownField> known;
  std::size_t source_root = 0, rhs_root = 0, source_end = 0;
  bool algebraic = false;
  unsigned timeout_ms = 1000;
};
}  // namespace tepl::core::inference
