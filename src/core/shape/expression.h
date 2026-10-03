#pragma once

#include <optional>

#include "src/ast/ast.h"
#include "src/core/shape/check_context.h"

namespace tepl::core::shape::detail {

class ExpressionChecker {
 public:
  explicit ExpressionChecker(CheckContext& context) : context_(context) {}
  ExprPtr check(const ast::ShapeExpr& expression,
                std::optional<TypeId> expected = std::nullopt);

 private:
  template <typename Value>
  ExprPtr make(const SourceOrigin& origin, TypeId type, Value value) {
    return std::make_shared<Expr>(Expr{origin, type, std::move(value)});
  }
  TypeId concrete(Type::Kind kind, const SourceOrigin& origin,
                  std::size_t depth = 0) {
    return context_.types.concrete({kind, depth}, origin);
  }
  ExprPtr lower(const ast::NameRef&, const SourceOrigin&);
  ExprPtr lower(const ast::IntegerLiteral&, const SourceOrigin&);
  ExprPtr lower(const ast::BooleanLiteral&, const SourceOrigin&);
  ExprPtr lower(const ast::ShapeAttrs&, const SourceOrigin&);
  ExprPtr lower(const ast::ShapeCall&, const SourceOrigin&);
  ExprPtr lower(const ast::ShapeUnary&, const SourceOrigin&);
  ExprPtr lower(const ast::ShapeBinary&, const SourceOrigin&);
  ExprPtr lower(const ast::ShapeList&, const SourceOrigin&);
  ExprPtr lower(const ast::ShapeIndex&, const SourceOrigin&);
  ExprPtr lower(const ast::ShapeField&, const SourceOrigin&);
  ExprPtr lower(const ast::ShapeConditional&, const SourceOrigin&);
  ExprPtr lower(const ast::ShapeComprehension&, const SourceOrigin&);
  CheckContext& context_;
};

}  // namespace tepl::core::shape::detail
