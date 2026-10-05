#pragma once

#include <optional>
#include <utility>

#include "src/core/analysis_context.h"
#include "src/core/rule/expanded_rule.h"
#include "src/core/rule/scope.h"

namespace tepl::core::detail {

class ExpressionChecker {
 public:
  ExpressionChecker(AnalysisContext& context, const ExpandedRule& input,
                    const Rule& rule, const RuleScope& scope,
                    builtins::Context section = builtins::Context::kWhere)
      : context_(context),
        input_(input),
        rule_(rule),
        scope_(scope),
        section_(section) {}

  // Failed expressions return nullptr and report their source diagnostic.
  TypedExprPtr check(const ast::ConstraintExpr& expression,
                     std::optional<TypeId> expected = {});

 private:
  template <typename Value>
  TypedExprPtr make(const SourceOrigin& origin, TypeId type, Value value,
                    std::optional<TypeId> expected = {}) {
    if (expected) context_.types.unify(type, *expected, origin);
    return std::make_shared<TypedExpr>(
        TypedExpr{origin, type, std::move(value)});
  }

  TypedExprPtr number(std::string spelling, bool decimal,
                      const SourceOrigin& origin,
                      std::optional<TypeId> expected);
  TypedExprPtr lower(const ast::NameRef&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const ast::AttributeRef&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const ast::DescriptorField&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const ast::IntegerLiteral&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const ast::FloatLiteral&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const ast::BooleanLiteral&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const ast::Call&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const ast::UnaryExpr&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const ast::BinaryExpr&, const SourceOrigin&,
                     std::optional<TypeId>);

  TypedExprPtr builtin(const ast::Call&, const SourceOrigin&,
                       std::optional<TypeId>);

  AnalysisContext& context_;
  const ExpandedRule& input_;
  const Rule& rule_;
  const RuleScope& scope_;
  builtins::Context section_;
};

}  // namespace tepl::core::detail
