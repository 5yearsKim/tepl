#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include "CLI/CLI.hpp"
#include "src/ast/print.h"
#include "src/core/analyze.h"
#include "src/core/print.h"
#include "src/host_codegen.h"
#include "src/parse.h"
#include "src/semantic.h"

int main(int argc, char** argv) {
  CLI::App app{"Parse TEPL tensor rewrite rules"};
  app.require_subcommand(1);
  std::string filename;
  bool print_tree = false;
  bool print_ast = false;
  bool implementation = false;
  auto* parse = app.add_subcommand("parse", "Validate a TEPL file");
  parse->add_option("file", filename, "TEPL input file")->required();
  auto* tree_option =
      parse->add_flag("--tree", print_tree, "Print the ANTLR parse tree");
  parse->add_flag("--ast", print_ast, "Print the TEPL AST")
      ->excludes(tree_option);
  auto* host_template = app.add_subcommand(
      "host-template", "Generate host function signatures from a TEPL file");
  host_template->add_option("file", filename, "TEPL input file")->required();
  host_template->add_flag("--impl", implementation,
                          "Generate a user implementation template");
  auto* check = app.add_subcommand(
      "check", "Analyze a TEPL file and print its checked IR");
  check->add_option("file", filename, "TEPL input file")->required();
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& error) {
    // Keep the CLI's existing exit code for usage errors.
    return app.exit(error) == 0 ? 0 : 2;
  }

  std::ifstream input(filename, std::ios::binary);
  if (!input) {
    std::cerr << filename << ": cannot open file\n";
    return 2;
  }
  std::string source{std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>()};
  if (input.bad()) {
    std::cerr << filename << ": cannot read file\n";
    return 2;
  }

  auto result = tepl::parse(source, filename);
  if (result.ok()) {
    auto import_diagnostics = tepl::resolveImports(*result.program);
    result.diagnostics.insert(result.diagnostics.end(),
                              import_diagnostics.begin(),
                              import_diagnostics.end());
    if (result.ok() && !*check) {
      const auto check_types = [&](const tepl::ast::Rule& rule) {
        for (const auto& diagnostic : tepl::validateTensorTypes(rule)) {
          result.diagnostics.push_back({diagnostic.span.begin.line,
                                        diagnostic.span.begin.column,
                                        diagnostic.message, rule.source_name});
        }
      };
      for (const auto& rule : result.program->rules) check_types(rule);
      for (const auto& rule : result.program->imported_rules) check_types(rule);
      auto dialect_diagnostics = tepl::validateDialectUses(*result.program);
      result.diagnostics.insert(result.diagnostics.end(),
                                dialect_diagnostics.begin(),
                                dialect_diagnostics.end());
    }
  }
  for (const auto& diagnostic : result.diagnostics) {
    std::cerr << (diagnostic.source_name.empty() ? filename
                                                 : diagnostic.source_name)
              << ':' << diagnostic.line << ':' << diagnostic.column << ": "
              << diagnostic.message << '\n';
  }
  if (!result.ok()) {
    return 1;
  }
  if (*check) {
    auto analyzed = tepl::core::analyze(*result.program);
    for (const auto& diagnostic : analyzed.diagnostics) {
      const auto print_location = [](const tepl::SourceLocation& at) {
        std::cerr << at.source_name << ':' << at.span.begin.line << ':'
                  << at.span.begin.column;
      };
      print_location(diagnostic.origin.definition);
      std::cerr << ": " << diagnostic.message << '\n';
      for (const auto& at : diagnostic.origin.expansions) {
        print_location(at);
        std::cerr << ": note: instantiated here\n";
      }
      for (const auto& at : diagnostic.related) {
        if (at.source_name.empty()) continue;
        print_location(at);
        std::cerr << ": note: related declaration or expression\n";
      }
    }
    if (!analyzed.ok()) return 1;
    std::cout << tepl::core::formatProgram(*analyzed.program);
    return 0;
  }
  if (*host_template) {
    auto generated =
        tepl::generateHostTemplate(*result.program, implementation);
    for (const auto& diagnostic : generated.diagnostics) {
      std::cerr << filename << ':' << diagnostic.span.begin.line << ':'
                << diagnostic.span.begin.column << ": " << diagnostic.message
                << '\n';
    }
    if (!generated.ok()) return 1;
    std::cout << generated.source;
    return 0;
  }
  if (print_tree) {
    std::cout << result.tree << '\n';
  } else if (print_ast) {
    std::cout << tepl::formatAst(*result.program);
  } else {
    std::cout << "Parsed " << result.rule_count << " rule(s).\n";
  }
}
