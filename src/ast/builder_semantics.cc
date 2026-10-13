#include "src/ast/builder.h"
#include "src/ast/builder_detail.h"

namespace tepl {
namespace {
using P = tepl_generated::TeplParser;
using E = ast::SemanticExpr;
using Ptr = ast::SemanticExprPtr;
}  // namespace
std::any AstBuilder::visitSemanticsProperty(P::SemanticsPropertyContext* ctx) {
  return ast::SemanticDefinition{
      detail::getSpan(ctx), std::any_cast<Ptr>(visit(ctx->semanticExpr()))};
}
std::any AstBuilder::visitSemanticCallExpr(P::SemanticCallExprContext* ctx) {
  std::vector<Ptr> args;
  if (auto* arguments = ctx->semanticArguments())
    for (auto* arg : arguments->semanticExpr())
      args.push_back(std::any_cast<Ptr>(visit(arg)));
  return std::make_shared<E>(E{detail::getSpan(ctx), E::Kind::kCall,
                               ctx->ID()->getText(), std::move(args)});
}
std::any AstBuilder::visitSemanticNameExpr(P::SemanticNameExprContext* ctx) {
  return std::make_shared<E>(
      E{detail::getSpan(ctx), E::Kind::kName, ctx->ID()->getText(), {}});
}
}  // namespace tepl
