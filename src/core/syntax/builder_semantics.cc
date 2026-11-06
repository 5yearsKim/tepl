#include "src/core/syntax/builder.h"
#include "src/core/syntax/builder_detail.h"

namespace tepl {
namespace {
using P = tepl_generated::TeplParser;
using E = syntax::SemanticExpr;
using Ptr = syntax::SemanticExprPtr;
}  // namespace
std::any AstBuilder::visitSemanticsProperty(P::SemanticsPropertyContext* ctx) {
  return syntax::SemanticDefinition{
      detail::getSpan(ctx), std::any_cast<Ptr>(visit(ctx->semanticExpr()))};
}
std::any AstBuilder::visitSemanticCallExpr(P::SemanticCallExprContext* ctx) {
  std::vector<Ptr> args;
  if (auto* arguments = ctx->semanticArguments())
    for (auto* arg : arguments->semanticExpr())
      args.push_back(std::any_cast<Ptr>(visit(arg)));
  return std::make_shared<const E>(E{detail::getSpan(ctx), E::Kind::kCall,
                                     ctx->ID()->getText(), std::move(args)});
}
std::any AstBuilder::visitSemanticAttributeExpr(
    P::SemanticAttributeExprContext* ctx) {
  return std::make_shared<const E>(
      E{detail::getSpan(ctx), E::Kind::kAttribute, ctx->ID()->getText(), {}});
}
std::any AstBuilder::visitSemanticNameExpr(P::SemanticNameExprContext* ctx) {
  return std::make_shared<const E>(
      E{detail::getSpan(ctx), E::Kind::kName, ctx->ID()->getText(), {}});
}
std::any AstBuilder::visitSemanticIntegerExpr(
    P::SemanticIntegerExprContext* ctx) {
  return std::make_shared<const E>(
      E{detail::getSpan(ctx), E::Kind::kInteger, ctx->getText(), {}});
}
}  // namespace tepl
