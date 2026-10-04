#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include "CLI/CLI.hpp"
#include "src/ast/print.h"
#include "src/codegen/generate.h"
#include "src/codegen/write.h"
#include "src/core/analyze.h"
#include "src/core/print.h"
#include "src/imports.h"
#include "src/parse.h"

int main(int argc, char** argv) {
  CLI::App app{"Parse, check, and generate TEPL tensor rewrite rules"};
  app.require_subcommand(1);
  std::string filename;
  bool print_tree = false;
  bool print_ast = false;
  auto* parse =
      app.add_subcommand("parse", "Parse a TEPL file and load imports");
  parse
      ->add_option("file", filename,
                   "TEPL input file (or project directory for check/generate)")
      ->required();
  auto* tree_option =
      parse->add_flag("--tree", print_tree, "Print the ANTLR parse tree");
  parse->add_flag("--ast", print_ast, "Print the TEPL AST")
      ->excludes(tree_option);
  auto* check = app.add_subcommand(
      "check", "Analyze a TEPL file and print its checked IR");
  check
      ->add_option("file", filename,
                   "TEPL input file (or project directory for check/generate)")
      ->required();
  std::string target = "rust";
  std::string output_directory;
  tepl::codegen::WriteOptions write_options;
  tepl::codegen::Options generation_options;
  auto* generate = app.add_subcommand(
      "generate", "Generate code from checked TEPL rules and dialects");
  generate
      ->add_option("file", filename,
                   "TEPL input file (or project directory for check/generate)")
      ->required();
  generate->add_option("--target", target, "Output language")
      ->check(CLI::IsMember({"rust", "cpp", "python"}));
  generate
      ->add_option("-o,--out", output_directory, "Generated module directory")
      ->required();
  generate->add_flag("--check", write_options.check,
                     "Check generated output without changing files");
  generate->add_flag(
      "--format,!--no-format", write_options.format,
      "Format output with rustfmt or clang-format (default: on)");
  generate->add_option("--cpp-namespace", generation_options.cpp_namespace,
                       "Namespace for generated C++ (default: tepl_generated)");
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& error) {
    return app.exit(error) == 0 ? 0 : 2;
  }

  tepl::ParseResult result;
  try {
    if ((*generate || *check) && std::filesystem::is_directory(filename)) {
      result = tepl::loadProject(filename);
      generation_options.rules_root =
          (std::filesystem::absolute(filename) / "rules").string();
    } else {
      std::ifstream input(filename, std::ios::binary);
      if (!input) {
        std::cerr << filename << ": cannot open file\n";
        return 2;
      }
      std::string source{std::istreambuf_iterator<char>(input), {}};
      if (input.bad()) {
        std::cerr << filename << ": cannot read file\n";
        return 2;
      }
      result = tepl::parse(source, filename);
      if (result.ok()) {
        auto diagnostics = tepl::resolveImports(*result.program);
        result.diagnostics.insert(result.diagnostics.end(), diagnostics.begin(),
                                  diagnostics.end());
      }
    }
  } catch (const std::filesystem::filesystem_error& error) {
    std::cerr << error.what() << '\n';
    return 2;
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
  if (*check || *generate) {
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
    if (*check) {
      std::cout << tepl::core::formatProgram(*analyzed.program);
      return 0;
    }
    generation_options.target = target == "rust" ? tepl::codegen::Target::kRust
                                : target == "cpp"
                                    ? tepl::codegen::Target::kCpp
                                    : tepl::codegen::Target::kPython;
    write_options.target = generation_options.target;
    const auto generated =
        tepl::codegen::generate(*analyzed.program, generation_options);
    for (const auto& diagnostic : generated.diagnostics) {
      const auto& at = diagnostic.origin.definition;
      if (!at.source_name.empty()) {
        std::cerr << at.source_name << ':' << at.span.begin.line << ':'
                  << at.span.begin.column << ": ";
      }
      std::cerr << diagnostic.message << '\n';
    }
    if (!generated.ok()) return 1;
    try {
      const auto written = tepl::codegen::synchronize(
          generated.files, output_directory, write_options);
      if (write_options.check) {
        for (const auto& path : written.differences)
          std::cerr << "generated output differs: " << path << '\n';
        if (!written.current()) return 1;
        std::cout << "Generated output is current in " << output_directory
                  << ".\n";
        return 0;
      }
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      return 2;
    }
    std::cout << "Generated " << generated.files.size() << " file(s) in "
              << output_directory << ".\n";
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
