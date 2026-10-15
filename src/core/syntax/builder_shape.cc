#include <utility>

#include "src/core/syntax/builder.h"
#include "src/core/syntax/builder_detail.h"

namespace tepl {
namespace {
using Parser = tepl_generated::TeplParser;
using detail::getSpan;

template <typename Value>
syntax::MetadataExprPtr makeMetadataExpr(
    const antlr4::ParserRuleContext* context, Value value) {
  return std::make_shared<syntax::MetadataExpr>(
      syntax::MetadataExpr{getSpan(context), std::move(value)});
}
}  // namespace

std::any AstBuilder::visitAxesProperty(Parser::AxesPropertyContext* context) {
  syntax::AxisDefinition result;
  result.span = getSpan(context);
  for (auto* statement : context->axisStatement()) {
    if (auto* metadata = statement->shapeStatement()) {
      result.statements.push_back(
          std::any_cast<syntax::MetadataStatement>(visit(metadata)));
    } else {
      syntax::AxisBind bind{
          getSpan(statement), statement->ID(1)->getText(), {}};
      for (auto* binding : statement->axisBinding())
        bind.bindings.push_back({getSpan(binding), binding->ID()->getText(),
                                 std::any_cast<syntax::MetadataExprPtr>(
                                     visit(binding->shapeExpr()))});
      if (statement->shapeExpr())
        bind.domain = std::any_cast<syntax::MetadataExprPtr>(
            visit(statement->shapeExpr()));
      result.statements.push_back(std::move(bind));
    }
  }
  result.output = std::any_cast<syntax::MetadataExprPtr>(
      visit(context->shapeYield()->shapeExpr()));
  return result;
}

std::any AstBuilder::visitShapeProperty(Parser::ShapePropertyContext* context) {
  syntax::MetadataDefinition definition;
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
        std::any_cast<syntax::MetadataStatement>(visit(statement)));
  }
  auto* result = context->shapeYield();
  definition.result = {getSpan(result), std::any_cast<syntax::MetadataExprPtr>(
                                            visit(result->shapeExpr()))};
  return definition;
}

std::any AstBuilder::visitShapeLetStatement(
    Parser::ShapeLetStatementContext* context) {
  auto* name = context->ID()->getSymbol();
  return syntax::MetadataStatement{
      getSpan(context),
      syntax::MetadataLet{
          context->ID()->getText(), getSpan(name, name),
          std::any_cast<syntax::MetadataExprPtr>(visit(context->shapeExpr()))}};
}

std::any AstBuilder::visitShapeAssertStatement(
    Parser::ShapeAssertStatementContext* context) {
  return syntax::MetadataStatement{
      getSpan(context),
      syntax::MetadataAssert{
          std::any_cast<syntax::MetadataExprPtr>(visit(context->shapeExpr()))}};
}

std::any AstBuilder::visitShapeConditionalExpr(
    Parser::ShapeConditionalExprContext* context) {
  return makeMetadataExpr(
      context,
      syntax::MetadataConditional{
          std::any_cast<syntax::MetadataExprPtr>(visit(context->shapeExpr(0))),
          std::any_cast<syntax::MetadataExprPtr>(visit(context->shapeExpr(1))),
          std::any_cast<syntax::MetadataExprPtr>(
              visit(context->shapeExpr(2)))});
}

std::any AstBuilder::visitShapeSimpleExpr(
    Parser::ShapeSimpleExprContext* context) {
  return visit(context->shapeLogicalOr());
}

template <typename Context>
syntax::MetadataExprPtr AstBuilder::foldMetadataBinary(
    antlr4::ParserRuleContext* parent, const std::vector<Context*>& operands) {
  auto lhs = std::any_cast<syntax::MetadataExprPtr>(visit(operands.front()));
  for (std::size_t index = 1; index < operands.size(); ++index) {
    const auto spelling = parent->children[2 * index - 1]->getText();
    auto rhs = std::any_cast<syntax::MetadataExprPtr>(visit(operands[index]));
    const syntax::SourceSpan span{lhs->span.begin, rhs->span.end};
    if (spelling == "++") {
      lhs = std::make_shared<syntax::MetadataExpr>(syntax::MetadataExpr{
          span,
          syntax::MetadataCall{"concat", {std::move(lhs), std::move(rhs)}}});
      continue;
    }
    const auto op = detail::binaryOperator(spelling);
    lhs = std::make_shared<syntax::MetadataExpr>(syntax::MetadataExpr{
        span, syntax::MetadataBinary{op, std::move(lhs), std::move(rhs)}});
  }
  return lhs;
}

std::any AstBuilder::visitShapeLogicalOr(
    Parser::ShapeLogicalOrContext* context) {
  return foldMetadataBinary(context, context->shapeLogicalAnd());
}

std::any AstBuilder::visitShapeLogicalAnd(
    Parser::ShapeLogicalAndContext* context) {
  return foldMetadataBinary(context, context->shapeEquality());
}

std::any AstBuilder::visitShapeEquality(Parser::ShapeEqualityContext* context) {
  return foldMetadataBinary(context, context->shapeComparison());
}

std::any AstBuilder::visitShapeComparison(
    Parser::ShapeComparisonContext* context) {
  return foldMetadataBinary(context, context->shapeAdditive());
}

std::any AstBuilder::visitShapeAdditive(Parser::ShapeAdditiveContext* context) {
  return foldMetadataBinary(context, context->shapeMultiplicative());
}

std::any AstBuilder::visitShapeMultiplicative(
    Parser::ShapeMultiplicativeContext* context) {
  return foldMetadataBinary(context, context->shapeUnary());
}

std::any AstBuilder::visitShapeUnary(Parser::ShapeUnaryContext* context) {
  if (auto* postfix = context->shapePostfix()) return visit(postfix);
  return makeMetadataExpr(
      context, syntax::MetadataUnary{
                   detail::unaryOperator(context->getStart()->getText()),
                   std::any_cast<syntax::MetadataExprPtr>(
                       visit(context->shapeUnary()))});
}

std::any AstBuilder::visitShapePostfix(Parser::ShapePostfixContext* context) {
  auto value =
      std::any_cast<syntax::MetadataExprPtr>(visit(context->shapePrimary()));
  for (auto* suffix : context->shapeSuffix()) {
    const syntax::SourceSpan span{value->span.begin, getSpan(suffix).end};
    if (auto* field = dynamic_cast<Parser::ShapeFieldSuffixContext*>(suffix)) {
      value = std::make_shared<syntax::MetadataExpr>(syntax::MetadataExpr{
          span,
          syntax::MetadataField{std::move(value), field->ID()->getText()}});
    } else {
      auto* index = dynamic_cast<Parser::ShapeIndexSuffixContext*>(suffix);
      value = std::make_shared<syntax::MetadataExpr>(syntax::MetadataExpr{
          span, syntax::MetadataIndex{std::move(value),
                                      std::any_cast<syntax::MetadataExprPtr>(
                                          visit(index->shapeExpr()))}});
    }
  }
  return value;
}

std::any AstBuilder::visitShapeCallPrimary(
    Parser::ShapeCallPrimaryContext* context) {
  syntax::MetadataCall call{context->ID()->getText(), {}};
  if (auto* arguments = context->shapeArguments()) {
    for (auto* argument : arguments->shapeExpr()) {
      call.arguments.push_back(
          std::any_cast<syntax::MetadataExprPtr>(visit(argument)));
    }
  }
  return makeMetadataExpr(context, std::move(call));
}

std::any AstBuilder::visitShapeNamePrimary(
    Parser::ShapeNamePrimaryContext* context) {
  return makeMetadataExpr(context, syntax::NameRef{context->ID()->getText()});
}

std::any AstBuilder::visitShapeAttrsPrimary(
    Parser::ShapeAttrsPrimaryContext* context) {
  return makeMetadataExpr(context, syntax::MetadataAttrs{});
}

std::any AstBuilder::visitShapeIntegerPrimary(
    Parser::ShapeIntegerPrimaryContext* context) {
  return makeMetadataExpr(context,
                          syntax::IntegerLiteral{context->INT()->getText()});
}

std::any AstBuilder::visitShapeTruePrimary(
    Parser::ShapeTruePrimaryContext* context) {
  return makeMetadataExpr(context, syntax::BooleanLiteral{true});
}

std::any AstBuilder::visitShapeFalsePrimary(
    Parser::ShapeFalsePrimaryContext* context) {
  return makeMetadataExpr(context, syntax::BooleanLiteral{false});
}

std::any AstBuilder::visitShapeGroupPrimary(
    Parser::ShapeGroupPrimaryContext* context) {
  auto expression =
      std::any_cast<syntax::MetadataExprPtr>(visit(context->shapeExpr()));
  return makeMetadataExpr(context, expression->value);
}

std::any AstBuilder::visitShapeEmptyListPrimary(
    Parser::ShapeEmptyListPrimaryContext* context) {
  return makeMetadataExpr(context, syntax::MetadataList{});
}

std::any AstBuilder::visitShapeListPrimary(
    Parser::ShapeListPrimaryContext* context) {
  syntax::MetadataList list;
  for (auto* element : context->shapeExpr()) {
    list.elements.push_back(
        std::any_cast<syntax::MetadataExprPtr>(visit(element)));
  }
  return makeMetadataExpr(context, std::move(list));
}

std::any AstBuilder::visitShapeComprehensionPrimary(
    Parser::ShapeComprehensionPrimaryContext* context) {
  auto* name = context->ID()->getSymbol();
  return makeMetadataExpr(
      context,
      syntax::MetadataComprehension{
          std::any_cast<syntax::MetadataExprPtr>(visit(context->shapeExpr(0))),
          context->ID()->getText(), getSpan(name, name),
          std::any_cast<syntax::MetadataExprPtr>(
              visit(context->shapeExpr(1)))});
}

}  // namespace tepl
