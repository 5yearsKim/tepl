#include <stdexcept>
#include <string_view>
#include <utility>

#include "src/ast/builder.h"
#include "src/ast/builder_detail.h"

namespace tepl {
namespace {
using Parser = tepl_generated::TeplParser;
using detail::getSpan;

template <typename Value>
ast::GraphExprPtr makeGraph(const antlr4::ParserRuleContext* context,
                            Value value) {
  return std::make_shared<ast::GraphExpr>(
      ast::GraphExpr{getSpan(context), std::move(value)});
}

template <typename Value>
ast::ConstraintExprPtr makeConstraint(const antlr4::ParserRuleContext* context,
                                      Value value) {
  return std::make_shared<ast::ConstraintExpr>(
      ast::ConstraintExpr{getSpan(context), std::move(value)});
}

ast::BinaryOp binaryOperator(std::string_view spelling) {
  static constexpr std::pair<std::string_view, ast::BinaryOp> operators[] = {
      {"+", ast::BinaryOp::kAdd},           {"-", ast::BinaryOp::kSubtract},
      {"*", ast::BinaryOp::kMultiply},      {"/", ast::BinaryOp::kDivide},
      {"%", ast::BinaryOp::kRemainder},     {"<", ast::BinaryOp::kLess},
      {"<=", ast::BinaryOp::kLessEqual},    {">", ast::BinaryOp::kGreater},
      {">=", ast::BinaryOp::kGreaterEqual}, {"==", ast::BinaryOp::kEqual},
      {"!=", ast::BinaryOp::kNotEqual},     {"&&", ast::BinaryOp::kLogicalAnd},
      {"||", ast::BinaryOp::kLogicalOr},
  };
  for (auto [text, op] : operators) {
    if (spelling == text) return op;
  }
  throw std::logic_error("unsupported binary operator in parse tree");
}

ast::UnaryOp unaryOperator(std::string_view spelling) {
  if (spelling == "+") return ast::UnaryOp::kPlus;
  if (spelling == "-") return ast::UnaryOp::kNegate;
  if (spelling == "!") return ast::UnaryOp::kLogicalNot;
  throw std::logic_error("unsupported unary operator in parse tree");
}

}  // namespace

std::any AstBuilder::visitRuleDecl(Parser::RuleDeclContext* context) {
  ast::Rule rule;
  rule.span = getSpan(context);
  rule.name = context->ID()->getText();
  for (auto* declaration : context->shapeDecl()) {
    rule.declarations.push_back(
        std::any_cast<ast::Declaration>(visit(declaration)));
  }
  rule.lhs = std::any_cast<ast::GraphExprPtr>(visit(context->graphExpr(0)));
  rule.rhs = std::any_cast<ast::GraphExprPtr>(visit(context->graphExpr(1)));
  if (auto* where = context->whereBlock()) {
    for (auto* expression : where->constraintExpr()) {
      rule.conditions.push_back(
          std::any_cast<ast::ConstraintExprPtr>(visit(expression)));
    }
  }
  if (auto* derive = context->deriveBlock()) {
    const auto targets = derive->attrRef();
    const auto values = derive->constraintExpr();
    for (std::size_t index = 0; index < targets.size(); ++index) {
      rule.derivations.push_back(
          {getSpan(targets[index]->getStart(), values[index]->getStop()),
           ast::AttributeRef{targets[index]->ID()->getText()},
           std::any_cast<ast::ConstraintExprPtr>(visit(values[index]))});
    }
  }
  return rule;
}

std::any AstBuilder::visitTensorDecl(Parser::TensorDeclContext* context) {
  ast::TensorDecl tensor{context->ID()->getText(), {}};
  if (auto* elements = context->shapeElements()) {
    tensor.shape =
        std::any_cast<std::vector<ast::ShapeDimension>>(visit(elements));
  }
  return ast::Declaration{getSpan(context), std::move(tensor)};
}

std::any AstBuilder::visitScalarDecl(Parser::ScalarDeclContext* context) {
  return ast::Declaration{getSpan(context),
                          ast::ScalarDecl{context->ID()->getText()}};
}

std::any AstBuilder::visitShapeElements(Parser::ShapeElementsContext* context) {
  std::vector<ast::ShapeDimension> dimensions;
  for (auto* child : context->children) {
    if (auto* scalar = dynamic_cast<Parser::ScalarDimContext*>(child)) {
      dimensions.push_back(std::any_cast<ast::ShapeDimension>(visit(scalar)));
    } else if (auto* sequence =
                   dynamic_cast<Parser::SequenceDimContext*>(child)) {
      dimensions.push_back(std::any_cast<ast::ShapeDimension>(visit(sequence)));
    }
  }
  return dimensions;
}

std::any AstBuilder::visitScalarDim(Parser::ScalarDimContext* context) {
  if (auto* name = context->ID()) {
    return ast::ShapeDimension{getSpan(context),
                               ast::NamedDimension{name->getText()}};
  }
  return ast::ShapeDimension{getSpan(context), ast::WildcardDimension{}};
}

std::any AstBuilder::visitSequenceDim(Parser::SequenceDimContext* context) {
  ast::SequenceDimension sequence;
  if (auto* name = context->ID()) sequence.name = name->getText();
  return ast::ShapeDimension{getSpan(context), std::move(sequence)};
}

std::any AstBuilder::visitBareBindingGraph(
    Parser::BareBindingGraphContext* context) {
  return visit(context->binding());
}

std::any AstBuilder::visitVariableGraph(Parser::VariableGraphContext* context) {
  return makeGraph(context, ast::NameRef{context->ID()->getText()});
}

std::any AstBuilder::visitParenthesizedBindingGraph(
    Parser::ParenthesizedBindingGraphContext* context) {
  auto binding = std::any_cast<ast::GraphExprPtr>(visit(context->binding()));
  binding->span = getSpan(context);
  return binding;
}

std::any AstBuilder::visitGetGraph(Parser::GetGraphContext* context) {
  return makeGraph(
      context, ast::Projection{ast::IntegerLiteral{context->INT()->getText()},
                               std::any_cast<ast::GraphExprPtr>(
                                   visit(context->graphExpr()))});
}

std::any AstBuilder::visitOperatorGraph(Parser::OperatorGraphContext* context) {
  ast::Operator op{context->opRef()->getText(), std::nullopt, {}};
  if (auto* attribute = context->attribute()) {
    op.attribute = ast::AttributeRef{attribute->attrRef()->ID()->getText()};
  }
  for (auto* operand : context->graphExpr()) {
    op.operands.push_back(std::any_cast<ast::GraphExprPtr>(visit(operand)));
  }
  return makeGraph(context, std::move(op));
}

std::any AstBuilder::visitBinding(Parser::BindingContext* context) {
  return makeGraph(context, ast::Binding{ast::NameRef{context->ID()->getText()},
                                         std::any_cast<ast::GraphExprPtr>(
                                             visit(context->graphExpr()))});
}

std::any AstBuilder::visitConstraintExpr(
    Parser::ConstraintExprContext* context) {
  return visit(context->logicalOr());
}

template <typename Context>
ast::ConstraintExprPtr AstBuilder::foldBinary(
    antlr4::ParserRuleContext* parent, const std::vector<Context*>& operands) {
  auto lhs = std::any_cast<ast::ConstraintExprPtr>(visit(operands.front()));
  for (std::size_t index = 1; index < operands.size(); ++index) {
    const auto op = binaryOperator(parent->children[2 * index - 1]->getText());
    auto rhs = std::any_cast<ast::ConstraintExprPtr>(visit(operands[index]));
    const ast::SourceSpan span{lhs->span.begin, rhs->span.end};
    lhs = std::make_shared<ast::ConstraintExpr>(ast::ConstraintExpr{
        span, ast::BinaryExpr{op, std::move(lhs), std::move(rhs)}});
  }
  return lhs;
}

std::any AstBuilder::visitLogicalOr(Parser::LogicalOrContext* context) {
  return foldBinary(context, context->logicalAnd());
}

std::any AstBuilder::visitLogicalAnd(Parser::LogicalAndContext* context) {
  return foldBinary(context, context->equality());
}

std::any AstBuilder::visitEquality(Parser::EqualityContext* context) {
  return foldBinary(context, context->comparison());
}

std::any AstBuilder::visitComparison(Parser::ComparisonContext* context) {
  return foldBinary(context, context->additive());
}

std::any AstBuilder::visitAdditive(Parser::AdditiveContext* context) {
  return foldBinary(context, context->multiplicative());
}

std::any AstBuilder::visitMultiplicative(
    Parser::MultiplicativeContext* context) {
  return foldBinary(context, context->unary());
}

std::any AstBuilder::visitUnary(Parser::UnaryContext* context) {
  if (auto* primary = context->primary()) return visit(primary);
  return makeConstraint(
      context, ast::UnaryExpr{unaryOperator(context->getStart()->getText()),
                              std::any_cast<ast::ConstraintExprPtr>(
                                  visit(context->unary()))});
}

std::any AstBuilder::visitCallPrimary(Parser::CallPrimaryContext* context) {
  ast::Call call{context->ID()->getText(), {}};
  if (auto* arguments = context->arguments()) {
    for (auto* argument : arguments->constraintExpr()) {
      call.arguments.push_back(
          std::any_cast<ast::ConstraintExprPtr>(visit(argument)));
    }
  }
  return makeConstraint(context, std::move(call));
}

std::any AstBuilder::visitNamePrimary(Parser::NamePrimaryContext* context) {
  return makeConstraint(context, ast::NameRef{context->ID()->getText()});
}

std::any AstBuilder::visitAttributePrimary(
    Parser::AttributePrimaryContext* context) {
  return makeConstraint(context,
                        ast::AttributeRef{context->attrRef()->ID()->getText()});
}

std::any AstBuilder::visitIntegerPrimary(
    Parser::IntegerPrimaryContext* context) {
  return makeConstraint(context,
                        ast::IntegerLiteral{context->INT()->getText()});
}

std::any AstBuilder::visitTruePrimary(Parser::TruePrimaryContext* context) {
  return makeConstraint(context, ast::BooleanLiteral{true});
}

std::any AstBuilder::visitFalsePrimary(Parser::FalsePrimaryContext* context) {
  return makeConstraint(context, ast::BooleanLiteral{false});
}

std::any AstBuilder::visitGroupPrimary(Parser::GroupPrimaryContext* context) {
  auto expression =
      std::any_cast<ast::ConstraintExprPtr>(visit(context->constraintExpr()));
  expression->span = getSpan(context);
  return expression;
}

}  // namespace tepl
