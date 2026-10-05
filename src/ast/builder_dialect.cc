#include <algorithm>

#include "src/ast/builder.h"
#include "src/ast/builder_detail.h"

namespace tepl {
namespace {
using Parser = tepl_generated::TeplParser;
using detail::getSpan;
}  // namespace

std::any AstBuilder::visitDialectDecl(Parser::DialectDeclContext* context) {
  ast::Dialect dialect{getSpan(context), context->ID()->getText(), {}, {}};
  dialect.source_name = source_name_;
  for (auto* schema : context->attrsDecl()) {
    dialect.schemas.push_back(std::any_cast<ast::AttrSchema>(visit(schema)));
  }
  for (auto* op : context->opDecl()) {
    dialect.operations.push_back(std::any_cast<ast::OpDecl>(visit(op)));
  }
  return dialect;
}

std::any AstBuilder::visitAttrsDecl(Parser::AttrsDeclContext* context) {
  ast::AttrSchema schema{getSpan(context), context->ID()->getText(), {}};
  for (auto* field : context->attrField()) {
    schema.fields.push_back(std::any_cast<ast::AttrField>(visit(field)));
  }
  return schema;
}

std::any AstBuilder::visitOpDecl(Parser::OpDeclContext* context) {
  ast::OpDecl op{getSpan(context),
                 context->opName()->getText(),
                 {},
                 context->ID()->getText(),
                 std::nullopt,
                 std::nullopt};
  if (auto* operands = context->operandDecls()) {
    for (auto* operand : operands->operandDecl()) {
      op.operands.push_back(std::any_cast<ast::OperandDecl>(visit(operand)));
    }
    if (auto* variadic = operands->variadicOperand()) {
      op.operands.push_back(std::any_cast<ast::OperandDecl>(visit(variadic)));
    }
  }
  if (auto* properties = context->opProperties()) {
    const auto duplicate = [&](std::size_t count, const char* property) {
      if (count > 1) op.duplicate_properties.push_back(property);
    };
    duplicate(properties->dtypeProperty().size(), "dtype");
    duplicate(properties->attrsProperty().size(), "attrs");
    duplicate(properties->aliasProperty().size(), "alias");
    if (!properties->dtypeProperty().empty()) {
      auto* dtype = properties->dtypeProperty().front();
      ast::MetadataDefinition definition;
      definition.span = getSpan(dtype);
      if (auto* inputs = dtype->shapeInputs()) {
        for (auto* input : inputs->shapeInput())
          definition.parameters.push_back(
              {getSpan(input), input->ID()->getText(), false});
        if (auto* input = inputs->shapeVariadicInput())
          definition.parameters.push_back(
              {getSpan(input), input->ID()->getText(), true});
      }
      for (auto* statement : dtype->shapeStatement())
        definition.statements.push_back(
            std::any_cast<ast::MetadataStatement>(visit(statement)));
      auto* result = dtype->shapeYield();
      definition.result = {getSpan(result), std::any_cast<ast::MetadataExprPtr>(
                                                visit(result->shapeExpr()))};
      op.dtype_definition = std::move(definition);
    }
    if (auto* shape = properties->shapeProperty())
      op.shape_definition =
          std::any_cast<ast::MetadataDefinition>(visit(shape));
    if (!properties->aliasProperty().empty())
      op.alias = properties->aliasProperty().front()->ID()->getText();
    if (!properties->attrsProperty().empty()) {
      auto* property = properties->attrsProperty().front();
      if (auto* shared =
              dynamic_cast<Parser::SharedAttrsPropertyContext*>(property)) {
        op.attrs = ast::SharedAttrs{shared->ID()->getText()};
      } else if (auto* inline_attrs =
                     dynamic_cast<Parser::InlineAttrsPropertyContext*>(
                         property)) {
        ast::InlineAttrs attrs;
        for (auto* field : inline_attrs->attrField())
          attrs.fields.push_back(std::any_cast<ast::AttrField>(visit(field)));
        op.attrs = std::move(attrs);
      }
    }
  }
  return op;
}

std::any AstBuilder::visitOperandDecl(Parser::OperandDeclContext* context) {
  const auto ids = context->ID();
  return ast::OperandDecl{getSpan(context), ids[0]->getText(),
                          ids[1]->getText(), false};
}

std::any AstBuilder::visitVariadicOperand(
    Parser::VariadicOperandContext* context) {
  const auto ids = context->ID();
  return ast::OperandDecl{getSpan(context), ids[0]->getText(),
                          ids[1]->getText(), true};
}

std::any AstBuilder::visitAttrField(Parser::AttrFieldContext* context) {
  auto* type = context->attrType();
  const auto text = type->getText();
  return ast::AttrField{
      getSpan(context),
      context->ID()->getText(),
      type->ID()->getText(),
      static_cast<std::size_t>(std::count(text.begin(), text.end(), '[')),
      text.ends_with("?"),
      context->attrDefault() != nullptr};
}

}  // namespace tepl
