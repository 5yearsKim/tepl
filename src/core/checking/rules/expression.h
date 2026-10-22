#pragma once

#include <optional>
#include <utility>

#include "src/core/checking/context.h"
#include "src/core/checking/rules/scope.h"
#include "src/core/resolution/rules/expanded_rule.h"

namespace tepl::core::detail {

class ExpressionChecker {
 public:
  ExpressionChecker(CheckingContext& context, const ExpandedRule& input,
                    Rule& rule, const RuleScope& scope,
                    builtins::Context section = builtins::Context::kWhere)
      : context_(context),
        input_(input),
        rule_(rule),
        scope_(scope),
        section_(section) {}

  // Failed expressions return nullptr and report their source diagnostic.
  TypedExprPtr check(const syntax::ConstraintExpr& expression,
                     std::optional<TypeId> expected = {});

  void resetLocals() { locals_.clear(); }
  TypedExprPtr local(const std::string& name,
                     const syntax::ConstraintExpr& value,
                     const SourceOrigin& origin);

 private:
  std::map<std::string, LocalId> locals_;
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
  TypedExprPtr lower(const syntax::NameRef&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const syntax::NodeProperty&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const syntax::IntegerLiteral&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const syntax::FloatLiteral&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const syntax::BooleanLiteral&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const syntax::Call&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const syntax::UnaryExpr&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const syntax::BinaryExpr&, const SourceOrigin&,
                     std::optional<TypeId>);

  TypedExprPtr lower(const syntax::StringLiteral&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const syntax::ListExpr&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const syntax::IndexExpr&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const syntax::ConditionalExpr&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const syntax::ComprehensionExpr&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const syntax::DescriptorConstructor&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr lower(const syntax::LocalBinding&, const SourceOrigin&,
                     std::optional<TypeId>);
  TypedExprPtr builtin(const syntax::Call&, const SourceOrigin&,
                       std::optional<TypeId>);

  CheckingContext& context_;
  const ExpandedRule& input_;
  Rule& rule_;
  const RuleScope& scope_;
  builtins::Context section_;
};

}  // namespace tepl::core::detail
