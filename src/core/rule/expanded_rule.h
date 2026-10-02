#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/ast/ast.h"
#include "src/core/ids.h"
#include "src/source.h"

namespace tepl::core::detail {

struct LocatedDeclaration {
  ast::Declaration value;
  SourceOrigin origin;
  // Different layers may constrain the same capture; duplicate declarations
  // within one layer are errors.
  std::size_t layer = 0;
};

struct LocatedDerivation {
  std::string target;
  ast::ConstraintExprPtr value;
  SourceOrigin origin;
};

struct GraphMetadata {
  SourceOrigin origin;
  std::optional<OpId> operation;
};

struct ExpandedRule {
  std::string source_name;
  std::string name;
  std::unordered_map<const ast::Call*, HostFunctionId> bound_functions;
  SourceOrigin origin;
  ast::GraphExprPtr lhs;
  ast::GraphExprPtr rhs;
  std::vector<LocatedDeclaration> declarations;
  std::vector<ast::ConstraintExprPtr> conditions;
  std::vector<LocatedDerivation> derivations;
  // Keys refer to the owning expression trees above and live for this stage.
  std::unordered_map<const ast::GraphExpr*, GraphMetadata> graphs;
  std::unordered_map<const ast::ConstraintExpr*, SourceOrigin> expr_origins;
};

}  // namespace tepl::core::detail
