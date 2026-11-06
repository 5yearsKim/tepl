#pragma once

#include <optional>

#include "src/core/checking/metadata/check_context.h"
#include "src/core/syntax/ast.h"

namespace tepl::core::metadata::detail {

class ExpressionChecker {
 public:
  explicit ExpressionChecker(CheckContext& context) : context_(context) {}
  ExprPtr check(const syntax::MetadataExpr& expression,
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
  ExprPtr lower(const syntax::NameRef&, const SourceOrigin&);
  ExprPtr lower(const syntax::IntegerLiteral&, const SourceOrigin&);
  ExprPtr lower(const syntax::BooleanLiteral&, const SourceOrigin&);
  ExprPtr lower(const syntax::MetadataAttrs&, const SourceOrigin&);
  ExprPtr lower(const syntax::MetadataCall&, const SourceOrigin&);
  ExprPtr lower(const syntax::MetadataUnary&, const SourceOrigin&);
  ExprPtr lower(const syntax::MetadataBinary&, const SourceOrigin&);
  ExprPtr lower(const syntax::MetadataList&, const SourceOrigin&);
  ExprPtr lower(const syntax::MetadataIndex&, const SourceOrigin&);
  ExprPtr lower(const syntax::MetadataField&, const SourceOrigin&);
  ExprPtr lower(const syntax::MetadataConditional&, const SourceOrigin&);
  ExprPtr lower(const syntax::MetadataComprehension&, const SourceOrigin&);
  CheckContext& context_;
};

}  // namespace tepl::core::metadata::detail
