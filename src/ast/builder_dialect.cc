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
                 std::nullopt,
                 {}};
  if (auto* operands = context->operandDecls()) {
    for (auto* operand : operands->operandDecl()) {
      op.operands.push_back(std::any_cast<ast::OperandDecl>(visit(operand)));
    }
    if (auto* variadic = operands->variadicOperand()) {
      op.operands.push_back(std::any_cast<ast::OperandDecl>(visit(variadic)));
    }
  }
  for (auto* property : context->opProperty()) {
    if (auto* alias = dynamic_cast<Parser::AliasPropertyContext*>(property)) {
      op.alias = alias->ID()->getText();
    } else if (auto* shared = dynamic_cast<Parser::SharedAttrsPropertyContext*>(
                   property)) {
      op.shared_attrs = shared->ID()->getText();
    } else if (auto* attrs = dynamic_cast<Parser::InlineAttrsPropertyContext*>(
                   property)) {
      for (auto* field : attrs->attrField()) {
        op.attrs.push_back(std::any_cast<ast::AttrField>(visit(field)));
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
  const auto ids = context->ID();
  return ast::AttrField{getSpan(context), ids[0]->getText(), ids[1]->getText(),
                        context->getText().find('[') != std::string::npos,
                        context->getText().find('=') != std::string::npos};
}

}  // namespace tepl
