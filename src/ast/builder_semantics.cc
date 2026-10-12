#include "src/ast/builder.h"
#include "src/ast/builder_detail.h"

namespace tepl {
namespace {
using P = tepl_generated::TeplParser;
using detail::getSpan;
using E = ast::SemanticExpr;
using Ptr = ast::SemanticExprPtr;
Ptr node(const antlr4::ParserRuleContext* ctx, E::Kind kind,
         std::string text = {}, std::vector<Ptr> args = {}) {
  return std::make_shared<E>(
      E{getSpan(ctx), kind, std::move(text), std::move(args), {}});
}
}  // namespace
std::any AstBuilder::visitSemanticsProperty(P::SemanticsPropertyContext* ctx) {
  ast::SemanticDefinition out{
      getSpan(ctx), ctx->ID().size() > 1 ? ctx->ID(1)->getText() : "", {}, {}};
  for (auto* let : ctx->semanticLet())
    out.locals.push_back({getSpan(let), let->ID()->getText(),
                          std::any_cast<Ptr>(visit(let->semanticExpr()))});
  out.result = std::any_cast<Ptr>(visit(ctx->semanticExpr()));
  return out;
}
std::any AstBuilder::visitSemanticConditionalExpr(
    P::SemanticConditionalExprContext* ctx) {
  return node(ctx, E::Kind::kConditional, "if",
              {std::any_cast<Ptr>(visit(ctx->semanticExpr(0))),
               std::any_cast<Ptr>(visit(ctx->semanticExpr(1))),
               std::any_cast<Ptr>(visit(ctx->semanticExpr(2)))});
}
std::any AstBuilder::visitSemanticSimpleExpr(
    P::SemanticSimpleExprContext* ctx) {
  return visit(ctx->semanticLogicalOr());
}
template <class C>
Ptr AstBuilder::foldSemantic(antlr4::ParserRuleContext* ctx,
                             const std::vector<C*>& args) {
  auto out = std::any_cast<Ptr>(visit(args.front()));
  for (std::size_t i = 1; i < args.size(); ++i)
    out = node(ctx, E::Kind::kBinary, ctx->children[2 * i - 1]->getText(),
               {out, std::any_cast<Ptr>(visit(args[i]))});
  return out;
}
std::any AstBuilder::visitSemanticLogicalOr(P::SemanticLogicalOrContext* ctx) {
  return foldSemantic(ctx, ctx->semanticLogicalAnd());
}
std::any AstBuilder::visitSemanticLogicalAnd(
    P::SemanticLogicalAndContext* ctx) {
  return foldSemantic(ctx, ctx->semanticEquality());
}
std::any AstBuilder::visitSemanticEquality(P::SemanticEqualityContext* ctx) {
  return foldSemantic(ctx, ctx->semanticComparison());
}
std::any AstBuilder::visitSemanticComparison(
    P::SemanticComparisonContext* ctx) {
  return foldSemantic(ctx, ctx->semanticAdditive());
}
std::any AstBuilder::visitSemanticAdditive(P::SemanticAdditiveContext* ctx) {
  return foldSemantic(ctx, ctx->semanticMultiplicative());
}
std::any AstBuilder::visitSemanticMultiplicative(
    P::SemanticMultiplicativeContext* ctx) {
  return foldSemantic(ctx, ctx->semanticUnary());
}
std::any AstBuilder::visitSemanticUnary(P::SemanticUnaryContext* ctx) {
  if (ctx->semanticPostfix()) return visit(ctx->semanticPostfix());
  return node(ctx, E::Kind::kUnary, ctx->getStart()->getText(),
              {std::any_cast<Ptr>(visit(ctx->semanticUnary()))});
}
std::any AstBuilder::visitSemanticPostfix(P::SemanticPostfixContext* ctx) {
  auto out = std::any_cast<Ptr>(visit(ctx->semanticPrimary()));
  for (auto* suffix : ctx->semanticSuffix()) {
    if (auto* field = dynamic_cast<P::SemanticFieldSuffixContext*>(suffix))
      out = node(ctx, E::Kind::kField, field->ID()->getText(), {out});
    else {
      auto* index = dynamic_cast<P::SemanticIndexSuffixContext*>(suffix);
      out = node(ctx, E::Kind::kIndex, "index",
                 {out, std::any_cast<Ptr>(visit(index->semanticExpr()))});
    }
  }
  return out;
}
std::any AstBuilder::visitSemanticCallPrimary(
    P::SemanticCallPrimaryContext* ctx) {
  std::vector<Ptr> args;
  if (auto* arguments = ctx->semanticArguments())
    for (auto* arg : arguments->semanticExpr())
      args.push_back(std::any_cast<Ptr>(visit(arg)));
  return node(ctx, E::Kind::kCall, ctx->ID()->getText(), std::move(args));
}
std::any AstBuilder::visitSemanticSumPrimary(
    P::SemanticSumPrimaryContext* ctx) {
  auto out = node(ctx, E::Kind::kSum, ctx->ID(1)->getText(),
                  {std::any_cast<Ptr>(visit(ctx->semanticExpr(0))),
                   std::any_cast<Ptr>(visit(ctx->semanticExpr(1)))});
  for (auto* let : ctx->semanticLet())
    out->locals.push_back({getSpan(let), let->ID()->getText(),
                           std::any_cast<Ptr>(visit(let->semanticExpr()))});
  return out;
}
std::any AstBuilder::visitSemanticNamePrimary(
    P::SemanticNamePrimaryContext* ctx) {
  return node(ctx, E::Kind::kName, ctx->ID()->getText());
}
std::any AstBuilder::visitSemanticAttrsPrimary(
    P::SemanticAttrsPrimaryContext* ctx) {
  return node(ctx, E::Kind::kAttrs, "attrs");
}
std::any AstBuilder::visitSemanticIntegerPrimary(
    P::SemanticIntegerPrimaryContext* ctx) {
  return node(ctx, E::Kind::kInteger, ctx->INT()->getText());
}
std::any AstBuilder::visitSemanticTruePrimary(
    P::SemanticTruePrimaryContext* ctx) {
  return node(ctx, E::Kind::kBool, "true");
}
std::any AstBuilder::visitSemanticFalsePrimary(
    P::SemanticFalsePrimaryContext* ctx) {
  return node(ctx, E::Kind::kBool, "false");
}
std::any AstBuilder::visitSemanticGroupPrimary(
    P::SemanticGroupPrimaryContext* ctx) {
  return visit(ctx->semanticExpr());
}
std::any AstBuilder::visitSemanticEmptyListPrimary(
    P::SemanticEmptyListPrimaryContext* ctx) {
  return node(ctx, E::Kind::kList);
}
std::any AstBuilder::visitSemanticListPrimary(
    P::SemanticListPrimaryContext* ctx) {
  std::vector<Ptr> args;
  for (auto* arg : ctx->semanticExpr())
    args.push_back(std::any_cast<Ptr>(visit(arg)));
  return node(ctx, E::Kind::kList, {}, std::move(args));
}
std::any AstBuilder::visitSemanticComprehensionPrimary(
    P::SemanticComprehensionPrimaryContext* ctx) {
  return node(ctx, E::Kind::kComprehension, ctx->ID()->getText(),
              {std::any_cast<Ptr>(visit(ctx->semanticExpr(1))),
               std::any_cast<Ptr>(visit(ctx->semanticExpr(0)))});
}
}  // namespace tepl
