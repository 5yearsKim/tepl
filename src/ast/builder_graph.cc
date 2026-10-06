#include "src/ast/builder.h"
#include "src/ast/builder_detail.h"

namespace tepl {
namespace {
using Parser = tepl_generated::TeplParser;
using detail::getSpan;
std::string unquote(const std::string& spelling) {
  std::string value;
  for (std::size_t i = 1; i + 1 < spelling.size(); ++i) {
    if (spelling[i] == '\\') ++i;
    value += spelling[i];
  }
  return value;
}
}  // namespace
ast::ConcreteValue AstBuilder::concreteValue(
    Parser::ConcreteValueContext* ctx) {
  using Kind = ast::ConcreteValue::Kind;
  ast::ConcreteValue value{Kind::kList, getSpan(ctx), ctx->getText(), {}};
  if (ctx->INT())
    value.kind = Kind::kInteger;
  else if (ctx->FLOAT())
    value.kind = Kind::kDecimal;
  else if (ctx->TRUE() || ctx->FALSE())
    value.kind = Kind::kBool;
  else if (ctx->STRING()) {
    value.kind = Kind::kString;
    value.text = unquote(value.text);
  } else if (ctx->ID())
    value.kind = Kind::kName;
  else
    for (auto* child : ctx->concreteValue())
      value.elements.push_back(concreteValue(child));
  return value;
}
ast::ConcreteExpr AstBuilder::concreteExpression(
    Parser::ConcreteExprContext* ctx) {
  using Kind = ast::ConcreteExpr::Kind;
  ast::ConcreteExpr expr{Kind::kReference, getSpan(ctx), ctx->getText()};
  if (ctx->opRef()) {
    expr.kind = Kind::kOperation;
    expr.name = ctx->opRef()->getText();
    if (auto* attributes = ctx->concreteAttributes())
      for (auto* field : attributes->concreteField())
        expr.attributes.push_back({getSpan(field), field->ID()->getText(),
                                   concreteValue(field->concreteValue())});
    for (auto* operand : ctx->concreteExpr())
      expr.operands.push_back(concreteExpression(operand));
  } else if (ctx->INT() || ctx->FLOAT()) {
    expr.kind = Kind::kLiteral;
    expr.decimal = ctx->FLOAT() != nullptr;
    if (auto* dtype = ctx->dtypeName()) {
      expr.dtype = ast::DTypeAnnotation{getSpan(dtype), dtype->getText()};
      expr.name.resize(expr.name.find(':'));
    }
  }
  return expr;
}
std::any AstBuilder::visitConcreteGraphDecl(
    Parser::ConcreteGraphDeclContext* ctx) {
  ast::ConcreteGraph graph;
  graph.span = getSpan(ctx);
  graph.name = ctx->ID(1)->getText();
  graph.source_name = source_name_;
  for (auto* declaration : ctx->concreteInput()) {
    ast::ConcreteInput input{getSpan(declaration),
                             declaration->ID(1)->getText()};
    if (auto* dtype = declaration->dtypeName())
      input.dtype = ast::DTypeAnnotation{getSpan(dtype), dtype->getText()};
    if (auto* shape = declaration->concreteShape()) {
      input.shape.emplace();
      for (auto* dimension : shape->INT())
        input.shape->push_back(dimension->getText());
    }
    graph.inputs.push_back(std::move(input));
  }
  for (auto* binding : ctx->concreteLet())
    graph.bindings.push_back({getSpan(binding), binding->ID()->getText(),
                              concreteExpression(binding->concreteExpr())});
  graph.output = concreteExpression(ctx->concreteExpr());
  return graph;
}
}  // namespace tepl
