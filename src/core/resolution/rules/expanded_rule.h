#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/syntax/ast.h"
#include "src/shared/catalog/ids.h"
#include "src/shared/source/source.h"

namespace tepl::core::detail {

struct LocatedDeclaration {
  syntax::Declaration value;
  SourceOrigin origin;
  // Different layers may constrain the same capture; duplicate declarations
  // within one layer are errors.
  std::size_t layer = 0;
};

struct LocatedDTypeDeclaration {
  syntax::DTypeDeclaration value;
  SourceOrigin origin;
  std::size_t layer;
};

struct LocatedDerivation {
  std::string target;
  syntax::ConstraintExprPtr value;
  SourceOrigin origin;
  bool local = false;
};

struct GraphMetadata {
  SourceOrigin origin;
  std::optional<OpId> operation;
};

struct ExpandedRule {
  std::string source_name;
  std::string name;
  SourceOrigin origin;
  syntax::GraphExprPtr lhs;
  syntax::GraphExprPtr rhs;
  std::vector<LocatedDeclaration> declarations;
  std::vector<LocatedDTypeDeclaration> dtypes;
  std::vector<syntax::ConstraintExprPtr> conditions;
  std::vector<std::size_t> condition_scopes;
  std::vector<LocatedDerivation> derivations;
  // Stable node IDs refer to the owning resolved expression trees above.
  std::unordered_map<std::size_t, GraphMetadata> graphs;
  std::unordered_map<std::size_t, SourceOrigin> expr_origins;
};

}  // namespace tepl::core::detail
