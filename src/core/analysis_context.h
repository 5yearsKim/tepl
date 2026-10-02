#pragma once

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/ast/ast.h"
#include "src/core/diagnostic.h"
#include "src/core/ir.h"
#include "src/core/type_inference.h"

namespace tepl::core::detail {

struct Scope {
  std::unordered_map<std::string, OpId> operations;
  std::unordered_map<std::string, const ast::Rule*> templates;
};

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
  std::string name;
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

struct AnalysisContext {
  explicit AnalysisContext(const ast::Program& input)
      : input(input), types(diagnostics) {}
  AnalysisContext(const AnalysisContext&) = delete;
  AnalysisContext& operator=(const AnalysisContext&) = delete;
  const ast::Program& input;
  Program output;
  std::vector<Diagnostic> diagnostics;
  std::map<std::string, Scope> scopes;
  std::unordered_map<std::string, HostFunctionId> host_names;
  TypeInference types;

  void report(SourceOrigin origin, std::string message,
              std::vector<SourceLocation> related = {});
  HostFunctionId host(const std::string& name, std::size_t arity,
                      const SourceOrigin& origin);
  std::optional<OpId> operation(const std::string& name,
                                const std::string& source,
                                const SourceOrigin& origin);
};

std::string sourceKey(const std::string& name);
std::string importTarget(const std::string& source, const std::string& path);
SourceOrigin origin(const std::string& source, ast::SourceSpan span);
void resolve(AnalysisContext& context);
std::optional<ExpandedRule> expand(AnalysisContext& context,
                                   const ast::Rule& rule);
std::optional<Rule> check(AnalysisContext& context, const ExpandedRule& input);

}  // namespace tepl::core::detail
