#include "src/semantic.h"

#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

namespace tepl {
namespace {

using Names = std::unordered_set<std::string>;

void checkGraph(const ast::GraphExpr& expression, bool lhs, Names& captures,
                std::vector<SemanticDiagnostic>& diagnostics) {
  if (const auto* name = std::get_if<ast::NameRef>(&expression.value)) {
    if (lhs) {
      captures.insert(name->name);
    } else if (!captures.contains(name->name)) {
      diagnostics.push_back(
          {expression.span, "unknown RHS variable '" + name->name + "'"});
    }
    return;
  }

  if (const auto* op = std::get_if<ast::Operator>(&expression.value)) {
    if (op->attribute) {
      diagnostics.push_back(
          {expression.span,
           "operator attributes are not supported by simple rules"});
    }
    for (const auto& operand : op->operands) {
      checkGraph(*operand, lhs, captures, diagnostics);
    }
    return;
  }

  diagnostics.push_back(
      {expression.span,
       "binders and projections are not supported by simple rules"});
}

}  // namespace

std::vector<SemanticDiagnostic> validateSimpleRule(const ast::Rule& rule) {
  std::vector<SemanticDiagnostic> diagnostics;
  for (const auto& declaration : rule.declarations) {
    diagnostics.push_back(
        {declaration.span,
         "shape declarations are not supported by simple rules"});
  }
  for (const auto& condition : rule.conditions) {
    diagnostics.push_back(
        {condition->span,
         "where conditions are not supported by simple rules"});
  }
  for (const auto& derivation : rule.derivations) {
    diagnostics.push_back(
        {derivation.span, "derive values are not supported by simple rules"});
  }

  Names captures;
  if (rule.lhs) {
    checkGraph(*rule.lhs, true, captures, diagnostics);
  } else {
    diagnostics.push_back({rule.span, "rule is missing its LHS"});
  }
  if (rule.rhs) {
    checkGraph(*rule.rhs, false, captures, diagnostics);
  } else {
    diagnostics.push_back({rule.span, "rule is missing its RHS"});
  }
  return diagnostics;
}

}  // namespace tepl
