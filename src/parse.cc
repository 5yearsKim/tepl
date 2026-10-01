#include "src/parse.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <unordered_map>
#include <unordered_set>

#include "antlr4-runtime.h"
#include "grammar/TeplLexer.h"
#include "grammar/TeplParser.h"
#include "src/ast/builder.h"

namespace tepl {
namespace {

class ErrorListener final : public antlr4::BaseErrorListener {
 public:
  explicit ErrorListener(std::vector<Diagnostic>& diagnostics)
      : diagnostics_(diagnostics) {}

  void syntaxError(antlr4::Recognizer*, antlr4::Token*, std::size_t line,
                   std::size_t column, const std::string& message,
                   std::exception_ptr) override {
    diagnostics_.push_back({line, column + 1, message});
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

ParseResult parse(std::string_view source, std::string_view source_name) {
  ParseResult result;
  ErrorListener errors(result.diagnostics);
  antlr4::ANTLRInputStream input{std::string(source)};
  tepl_generated::TeplLexer lexer(&input);
  lexer.removeErrorListeners();
  lexer.addErrorListener(&errors);
  antlr4::CommonTokenStream tokens(&lexer);
  tepl_generated::TeplParser parser(&tokens);
  parser.removeErrorListeners();
  parser.addErrorListener(&errors);

  auto* program = parser.program();
  if (result.ok()) {
    result.rule_count = program->ruleDecl().size();
    result.tree = program->toStringTree(&parser);
    result.program = AstBuilder{}.build(program, std::string(source_name));
  }
  return result;
}

std::vector<Diagnostic> resolveImports(ast::Program& program) {
  namespace fs = std::filesystem;
  std::vector<Diagnostic> diagnostics;
  std::unordered_set<std::string> active;
  std::unordered_set<std::string> loaded;
  std::unordered_set<std::string> dialect_keys;
  const auto key = [](const ast::Dialect& dialect) {
    return dialect.source_name + "\n" + dialect.name;
  };
  for (const auto& dialect : program.dialects)
    dialect_keys.insert(key(dialect));
  const auto root = fs::absolute(program.source_name).lexically_normal();
  active.insert(root.string());
  loaded.insert(root.string());

  std::function<void(const ast::Program&, const fs::path&)> load =
      [&](const ast::Program& current, const fs::path& file) {
        for (const auto& imported : current.imports) {
          const auto target = fs::absolute(file.parent_path() / imported.path)
                                  .lexically_normal();
          const auto report = [&](std::string message) {
            diagnostics.push_back({imported.span.begin.line,
                                   imported.span.begin.column,
                                   std::move(message), file.string()});
          };
          if (active.contains(target.string())) {
            report("cyclic import '" + imported.path + "'");
            continue;
          }
          std::ifstream input(target, std::ios::binary);
          if (!input) {
            report("cannot open import '" + imported.path + "'");
            continue;
          }
          std::string source{std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>()};
          if (input.bad()) {
            report("cannot read import '" + imported.path + "'");
            continue;
          }
          auto parsed = parse(source, target.string());
          if (!parsed.ok()) {
            for (const auto& diagnostic : parsed.diagnostics) {
              diagnostics.push_back({diagnostic.line, diagnostic.column,
                                     diagnostic.message, target.string()});
            }
            continue;
          }
          if (!parsed.program->rules.empty()) {
            report("import '" + imported.path +
                   "' contains rules; imports must define dialects only");
            continue;
          }
          std::unordered_set<std::string> direct_names;
          for (const auto& dialect : parsed.program->dialects) {
            if (!direct_names.insert(dialect.name).second) {
              diagnostics.push_back({dialect.span.begin.line,
                                     dialect.span.begin.column,
                                     "duplicate dialect '" + dialect.name + "'",
                                     target.string()});
            }
          }
          if (imported.dialect) {
            const bool found =
                std::any_of(parsed.program->dialects.begin(),
                            parsed.program->dialects.end(),
                            [&](const ast::Dialect& dialect) {
                              return dialect.name == *imported.dialect;
                            });
            if (!found) {
              report("dialect '" + *imported.dialect +
                     "' is not defined in import '" + imported.path + "'");
              continue;
            }
          }
          if (loaded.insert(target.string()).second) {
            active.insert(target.string());
            load(*parsed.program, target);
            active.erase(target.string());
          }
          for (auto& dialect : parsed.program->dialects) {
            if (imported.dialect && dialect.name != *imported.dialect) continue;
            if (dialect_keys.insert(key(dialect)).second) {
              program.dialects.push_back(std::move(dialect));
            }
          }
        }
      };
  load(program, root);
  return diagnostics;
}

std::vector<Diagnostic> validateDialectUses(const ast::Program& program) {
  std::vector<Diagnostic> diagnostics;
  if (program.dialects.empty() && program.uses.empty()) return diagnostics;
  using Operations = std::unordered_map<std::string, const ast::OpDecl*>;
  std::unordered_map<const ast::Dialect*, Operations> operations;
  std::unordered_set<std::string> dialect_keys;
  for (const auto& dialect : program.dialects) {
    const auto report = [&](const ast::SourceSpan& span, std::string message) {
      diagnostics.push_back({span.begin.line, span.begin.column,
                             std::move(message), dialect.source_name});
    };
    if (!dialect_keys.insert(dialect.source_name + "\n" + dialect.name)
             .second) {
      report(dialect.span, "duplicate dialect '" + dialect.name + "'");
    }
    auto& names = operations[&dialect];
    std::unordered_set<std::string> schemas;
    for (const auto& schema : dialect.schemas) {
      if (!schemas.insert(schema.name).second) {
        report(schema.span, "duplicate attribute schema '" + schema.name + "'");
      }
    }
    const auto checkFields = [&](const std::vector<ast::AttrField>& fields) {
      std::unordered_set<std::string> names;
      for (const auto& field : fields) {
        if (!names.insert(field.name).second) {
          report(field.span, "duplicate attribute field '" + field.name + "'");
        }
        if (field.type != "index" && field.type != "string") {
          report(field.span, "unknown attribute type '" + field.type + "'");
        }
        if (field.empty_default && !field.list) {
          report(field.span, "empty list default requires a list type");
        }
      }
    };
    for (const auto& schema : dialect.schemas) checkFields(schema.fields);
    for (const auto& op : dialect.operations) {
      if (!names.emplace(op.name, &op).second) {
        report(op.span, "duplicate operation name '" + op.name + "'");
      }
      if (op.alias && !names.emplace(*op.alias, &op).second) {
        report(op.span, "duplicate operation alias '" + *op.alias + "'");
      }
      if (op.result_type != "tensor") {
        report(op.span, "unknown result type '" + op.result_type + "'");
      }
      std::unordered_set<std::string> operand_names;
      for (const auto& operand : op.operands) {
        if (!operand_names.insert(operand.name).second) {
          report(operand.span, "duplicate operand '" + operand.name + "'");
        }
        if (operand.type != "tensor") {
          report(operand.span, "unknown operand type '" + operand.type + "'");
        }
      }
      if (op.attrs) {
        if (const auto* shared = std::get_if<ast::SharedAttrs>(&*op.attrs)) {
          if (!schemas.contains(shared->name)) {
            report(op.span, "unknown attribute schema '" + shared->name + "'");
          }
        } else {
          checkFields(std::get<ast::InlineAttrs>(*op.attrs).fields);
        }
      }
    }
  }
  const auto report = [&](const ast::SourceSpan& span, std::string message) {
    diagnostics.push_back(
        {span.begin.line, span.begin.column, std::move(message)});
  };
  std::unordered_map<std::string, const ast::Dialect*> aliases;
  Operations visible;
  const auto open = [&](std::string_view name, const ast::OpDecl* op,
                        const ast::SourceSpan& span) {
    const auto [found, inserted] = visible.emplace(name, op);
    if (!inserted && found->second != op) {
      report(span, "ambiguous operation '" + std::string(name) + "'");
    }
  };
  const auto openAll = [&](const ast::Dialect& dialect,
                           const ast::SourceSpan& span) {
    for (const auto& [name, op] : operations.at(&dialect)) {
      open(name, op, span);
    }
  };
  for (const auto& dialect : program.dialects) {
    if (dialect.source_name == program.source_name) {
      aliases.emplace(dialect.name, &dialect);
      openAll(dialect, dialect.span);
    }
  }
  for (const auto& imported : program.imports) {
    const auto target =
        std::filesystem::absolute(
            std::filesystem::path(program.source_name).parent_path() /
            imported.path)
            .lexically_normal()
            .string();
    for (const auto& dialect : program.dialects) {
      if (dialect.source_name != target) continue;
      if (imported.dialect) {
        if (dialect.name != *imported.dialect) continue;
        const auto& alias = *imported.alias;
        const auto [found, inserted] = aliases.emplace(alias, &dialect);
        if (!inserted && found->second != &dialect) {
          report(imported.span, "duplicate dialect alias '" + alias + "'");
        }
      } else {
        openAll(dialect, imported.span);
      }
    }
  }
  for (const auto& used : program.uses) {
    const auto found = aliases.find(used.alias);
    if (found == aliases.end()) {
      report(used.span, "unknown dialect alias '" + used.alias + "'");
      continue;
    }
    const auto& names = operations.at(found->second);
    if (used.operations.empty()) {
      openAll(*found->second, used.span);
    } else {
      for (const auto& name : used.operations) {
        const auto op = names.find(name);
        if (op == names.end()) {
          report(used.span, "unknown operation '" + name +
                                "' in dialect alias '" + used.alias + "'");
        } else {
          open(name, op->second, used.span);
        }
      }
    }
  }
  std::function<void(const ast::GraphExpr&)> check = [&](const ast::GraphExpr&
                                                             expression) {
    if (const auto* op = std::get_if<ast::Operator>(&expression.value)) {
      const ast::OpDecl* declaration = nullptr;
      bool unknown_alias = false;
      const auto dot = op->name.find('.');
      if (dot == std::string::npos) {
        const auto found = visible.find(op->name);
        if (found != visible.end()) declaration = found->second;
      } else {
        const auto alias = op->name.substr(0, dot);
        const auto found = aliases.find(alias);
        if (found == aliases.end()) {
          report(expression.span, "unknown dialect alias '" + alias + "'");
          unknown_alias = true;
        } else {
          const auto& names = operations.at(found->second);
          const auto operation = names.find(op->name.substr(dot + 1));
          if (operation != names.end()) declaration = operation->second;
        }
      }
      if (!declaration) {
        if (!unknown_alias) {
          report(expression.span, "unknown operation '" + op->name + "'");
        }
      } else {
        const bool variadic = !declaration->operands.empty() &&
                              declaration->operands.back().variadic;
        const std::size_t minimum =
            declaration->operands.size() - (variadic ? 1 : 0);
        if (op->operands.size() < minimum ||
            (!variadic && op->operands.size() != minimum)) {
          report(expression.span, "operation '" + op->name + "' expects " +
                                      (variadic ? "at least " : "exactly ") +
                                      std::to_string(minimum) +
                                      " operands, got " +
                                      std::to_string(op->operands.size()));
        }
        const bool needs_attrs =
            declaration->attrs &&
            (std::holds_alternative<ast::SharedAttrs>(*declaration->attrs) ||
             !std::get<ast::InlineAttrs>(*declaration->attrs).fields.empty());
        if (needs_attrs && !op->attribute) {
          report(expression.span, "operation '" + op->name +
                                      "' requires an attribute descriptor");
        } else if (!needs_attrs && op->attribute) {
          report(expression.span,
                 "operation '" + op->name + "' has no attributes");
        }
      }
      for (const auto& operand : op->operands) check(*operand);
    } else if (const auto* binding =
                   std::get_if<ast::Binding>(&expression.value)) {
      check(*binding->expression);
    } else if (const auto* projection =
                   std::get_if<ast::Projection>(&expression.value)) {
      check(*projection->tuple);
    }
  };
  for (const auto& rule : program.rules) {
    check(*rule.lhs);
    check(*rule.rhs);
  }
  return diagnostics;
}

}  // namespace tepl
