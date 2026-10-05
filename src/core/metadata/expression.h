#pragma once

#include <optional>

#include "src/ast/ast.h"
#include "src/core/metadata/check_context.h"

namespace tepl::core::metadata::detail {

class ExpressionChecker {
 public:
  explicit ExpressionChecker(CheckContext& context) : context_(context) {}
  ExprPtr check(const ast::MetadataExpr& expression,
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
  ExprPtr lower(const ast::MetadataAttrs&, const SourceOrigin&);
  ExprPtr lower(const ast::MetadataCall&, const SourceOrigin&);
  ExprPtr lower(const ast::MetadataUnary&, const SourceOrigin&);
  ExprPtr lower(const ast::MetadataBinary&, const SourceOrigin&);
  ExprPtr lower(const ast::MetadataList&, const SourceOrigin&);
  ExprPtr lower(const ast::MetadataIndex&, const SourceOrigin&);
  ExprPtr lower(const ast::MetadataField&, const SourceOrigin&);
  ExprPtr lower(const ast::MetadataConditional&, const SourceOrigin&);
  ExprPtr lower(const ast::MetadataComprehension&, const SourceOrigin&);
  CheckContext& context_;
};

}  // namespace tepl::core::metadata::detail
