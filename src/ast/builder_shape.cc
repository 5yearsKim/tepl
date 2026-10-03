#include <utility>

#include "src/ast/builder.h"
#include "src/ast/builder_detail.h"

namespace tepl {
namespace {
using Parser = tepl_generated::TeplParser;
using detail::getSpan;

template <typename Value>
ast::ShapeExprPtr makeShape(const antlr4::ParserRuleContext* context,
                            Value value) {
  return std::make_shared<ast::ShapeExpr>(
      ast::ShapeExpr{getSpan(context), std::move(value)});
}
}  // namespace

std::any AstBuilder::visitShapeProperty(Parser::ShapePropertyContext* context) {
  ast::ShapeDefinition definition;
  definition.span = getSpan(context);
  if (auto* inputs = context->shapeInputs()) {
    for (auto* input : inputs->shapeInput()) {
      definition.parameters.push_back(
          {getSpan(input), input->ID()->getText(), false});
    }
    if (auto* input = inputs->shapeVariadicInput()) {
      definition.parameters.push_back(
          {getSpan(input), input->ID()->getText(), true});
    }
  }
  for (auto* statement : context->shapeStatement()) {
    definition.statements.push_back(
        std::any_cast<ast::ShapeStatement>(visit(statement)));
  }
  auto* result = context->shapeYield();
  definition.result = {getSpan(result), std::any_cast<ast::ShapeExprPtr>(
                                            visit(result->shapeExpr()))};
  return definition;
}

std::any AstBuilder::visitShapeLetStatement(
    Parser::ShapeLetStatementContext* context) {
  auto* name = context->ID()->getSymbol();
  return ast::ShapeStatement{
      getSpan(context),
      ast::ShapeLet{
          context->ID()->getText(), getSpan(name, name),
          std::any_cast<ast::ShapeExprPtr>(visit(context->shapeExpr()))}};
}

std::any AstBuilder::visitShapeAssertStatement(
    Parser::ShapeAssertStatementContext* context) {
  return ast::ShapeStatement{getSpan(context),
                             ast::ShapeAssert{std::any_cast<ast::ShapeExprPtr>(
                                 visit(context->shapeExpr()))}};
}

std::any AstBuilder::visitShapeConditionalExpr(
    Parser::ShapeConditionalExprContext* context) {
  return makeShape(
      context,
      ast::ShapeConditional{
          std::any_cast<ast::ShapeExprPtr>(visit(context->shapeExpr(0))),
          std::any_cast<ast::ShapeExprPtr>(visit(context->shapeExpr(1))),
          std::any_cast<ast::ShapeExprPtr>(visit(context->shapeExpr(2)))});
}

std::any AstBuilder::visitShapeSimpleExpr(
    Parser::ShapeSimpleExprContext* context) {
  return visit(context->shapeLogicalOr());
}

template <typename Context>
ast::ShapeExprPtr AstBuilder::foldShapeBinary(
    antlr4::ParserRuleContext* parent, const std::vector<Context*>& operands) {
  auto lhs = std::any_cast<ast::ShapeExprPtr>(visit(operands.front()));
  for (std::size_t index = 1; index < operands.size(); ++index) {
    const auto op =
        detail::binaryOperator(parent->children[2 * index - 1]->getText());
    auto rhs = std::any_cast<ast::ShapeExprPtr>(visit(operands[index]));
    const ast::SourceSpan span{lhs->span.begin, rhs->span.end};
    lhs = std::make_shared<ast::ShapeExpr>(ast::ShapeExpr{
        span, ast::ShapeBinary{op, std::move(lhs), std::move(rhs)}});
  }
  return lhs;
}

std::any AstBuilder::visitShapeLogicalOr(
    Parser::ShapeLogicalOrContext* context) {
  return foldShapeBinary(context, context->shapeLogicalAnd());
}

std::any AstBuilder::visitShapeLogicalAnd(
    Parser::ShapeLogicalAndContext* context) {
  return foldShapeBinary(context, context->shapeEquality());
}

std::any AstBuilder::visitShapeEquality(Parser::ShapeEqualityContext* context) {
  return foldShapeBinary(context, context->shapeComparison());
}

std::any AstBuilder::visitShapeComparison(
    Parser::ShapeComparisonContext* context) {
  return foldShapeBinary(context, context->shapeAdditive());
}

std::any AstBuilder::visitShapeAdditive(Parser::ShapeAdditiveContext* context) {
  return foldShapeBinary(context, context->shapeMultiplicative());
}

std::any AstBuilder::visitShapeMultiplicative(
    Parser::ShapeMultiplicativeContext* context) {
  return foldShapeBinary(context, context->shapeUnary());
}

std::any AstBuilder::visitShapeUnary(Parser::ShapeUnaryContext* context) {
  if (auto* postfix = context->shapePostfix()) return visit(postfix);
  return makeShape(
      context,
      ast::ShapeUnary{
          detail::unaryOperator(context->getStart()->getText()),
          std::any_cast<ast::ShapeExprPtr>(visit(context->shapeUnary()))});
}

std::any AstBuilder::visitShapePostfix(Parser::ShapePostfixContext* context) {
  auto value = std::any_cast<ast::ShapeExprPtr>(visit(context->shapePrimary()));
  for (auto* suffix : context->shapeSuffix()) {
    const ast::SourceSpan span{value->span.begin, getSpan(suffix).end};
    if (auto* field = dynamic_cast<Parser::ShapeFieldSuffixContext*>(suffix)) {
      value = std::make_shared<ast::ShapeExpr>(ast::ShapeExpr{
          span, ast::ShapeField{std::move(value), field->ID()->getText()}});
    } else {
      auto* index = dynamic_cast<Parser::ShapeIndexSuffixContext*>(suffix);
      value = std::make_shared<ast::ShapeExpr>(ast::ShapeExpr{
          span,
          ast::ShapeIndex{std::move(value), std::any_cast<ast::ShapeExprPtr>(
                                                visit(index->shapeExpr()))}});
    }
  }
  return value;
}

std::any AstBuilder::visitShapeCallPrimary(
    Parser::ShapeCallPrimaryContext* context) {
  ast::ShapeCall call{context->ID()->getText(), {}};
  if (auto* arguments = context->shapeArguments()) {
    for (auto* argument : arguments->shapeExpr()) {
      call.arguments.push_back(
          std::any_cast<ast::ShapeExprPtr>(visit(argument)));
    }
  }
  return makeShape(context, std::move(call));
}

std::any AstBuilder::visitShapeNamePrimary(
    Parser::ShapeNamePrimaryContext* context) {
  return makeShape(context, ast::NameRef{context->ID()->getText()});
}

std::any AstBuilder::visitShapeAttrsPrimary(
    Parser::ShapeAttrsPrimaryContext* context) {
  return makeShape(context, ast::ShapeAttrs{});
}

std::any AstBuilder::visitShapeIntegerPrimary(
    Parser::ShapeIntegerPrimaryContext* context) {
  return makeShape(context, ast::IntegerLiteral{context->INT()->getText()});
}

std::any AstBuilder::visitShapeTruePrimary(
    Parser::ShapeTruePrimaryContext* context) {
  return makeShape(context, ast::BooleanLiteral{true});
}

std::any AstBuilder::visitShapeFalsePrimary(
    Parser::ShapeFalsePrimaryContext* context) {
  return makeShape(context, ast::BooleanLiteral{false});
}

std::any AstBuilder::visitShapeGroupPrimary(
    Parser::ShapeGroupPrimaryContext* context) {
  auto expression =
      std::any_cast<ast::ShapeExprPtr>(visit(context->shapeExpr()));
  expression->span = getSpan(context);
  return expression;
}

std::any AstBuilder::visitShapeEmptyListPrimary(
    Parser::ShapeEmptyListPrimaryContext* context) {
  return makeShape(context, ast::ShapeList{});
}

std::any AstBuilder::visitShapeListPrimary(
    Parser::ShapeListPrimaryContext* context) {
  ast::ShapeList list;
  for (auto* element : context->shapeExpr()) {
    list.elements.push_back(std::any_cast<ast::ShapeExprPtr>(visit(element)));
  }
  return makeShape(context, std::move(list));
}

std::any AstBuilder::visitShapeComprehensionPrimary(
    Parser::ShapeComprehensionPrimaryContext* context) {
  auto* name = context->ID()->getSymbol();
  return makeShape(
      context,
      ast::ShapeComprehension{
          std::any_cast<ast::ShapeExprPtr>(visit(context->shapeExpr(0))),
          context->ID()->getText(), getSpan(name, name),
          std::any_cast<ast::ShapeExprPtr>(visit(context->shapeExpr(1)))});
}

}  // namespace tepl
