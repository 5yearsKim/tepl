#include "src/ast/builder.h"

#include <utility>

#include "src/ast/builder_detail.h"

namespace tepl {
namespace {
using Parser = tepl_generated::TeplParser;
using detail::getSpan;
}  // namespace

ast::Program AstBuilder::build(Parser::ProgramContext* context,
                               std::string source_name) {
  source_name_ = std::move(source_name);
  return std::any_cast<ast::Program>(visit(context));
}

std::any AstBuilder::visitProgram(Parser::ProgramContext* context) {
  ast::Program program{getSpan(context), source_name_, {}};
  for (auto* imported : context->importDecl()) {
    program.imports.push_back(std::any_cast<ast::Import>(visit(imported)));
  }
  for (auto* used : context->useDecl()) {
    program.uses.push_back(std::any_cast<ast::Use>(visit(used)));
  }
  for (auto* dialect : context->dialectDecl()) {
    program.dialects.push_back(std::any_cast<ast::Dialect>(visit(dialect)));
  }
  for (auto* rule : context->ruleDecl()) {
    program.rules.push_back(std::any_cast<ast::Rule>(visit(rule)));
  }
  for (auto* graph : context->concreteGraphDecl())
    program.graphs.push_back(std::any_cast<ast::ConcreteGraph>(visit(graph)));
  return program;
}

std::any AstBuilder::visitImportDecl(Parser::ImportDeclContext* context) {
  const std::string spelling = context->STRING()->getText();
  std::string path;
  for (std::size_t index = 1; index + 1 < spelling.size(); ++index) {
    if (spelling[index] == '\\') ++index;
    path += spelling[index];
  }
  ast::Import imported{getSpan(context), std::move(path)};
  if (auto* names = context->ruleImportNames()) {
    for (auto* name : names->ID()) {
      imported.rules.push_back(ast::NameRef{name->getText()});
    }
  } else if (context->FROM()) {
    const auto names = context->ID();
    imported.dialect = names[0]->getText();
    imported.alias = names.size() == 2 ? names[1]->getText() : imported.dialect;
  }
  return imported;
}

std::any AstBuilder::visitUseDecl(Parser::UseDeclContext* context) {
  ast::Use used{getSpan(context), context->ID()->getText(), {}};
  for (auto* operation : context->opName()) {
    used.operations.push_back(operation->getText());
  }
  return used;
}

}  // namespace tepl
