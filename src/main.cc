#include <cstdlib>
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
#include "src/program/program.h"
#include "src/rewrite/cli.h"
#include "src/semantic_check.h"
#include "src/support/init.h"

int main(int argc, char** argv) {
  CLI::App app{"Compile, rewrite, and generate code for TEPL projects"};
  app.require_subcommand(1);
  std::string project_directory;
  auto* init = app.add_subcommand("init", "Create a minimal TEPL project");
  init->add_option("dir", project_directory, "Project directory to initialize")
      ->required();
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
      "generate",
      "Generate code from checked TEPL rules, dialects, and graphs");
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
  auto* compile =
      app.add_subcommand("compile", "Compile TEPL into portable rewrite JSON");
  compile->add_option("file", filename, "TEPL file or project")->required();
  compile->add_option("-o,--out", output_directory,
                      "Output JSON file (default: stdout)");
  auto* validate = app.add_subcommand(
      "validate", "Validate a compiled rewrite JSON program");
  validate->add_option("file", filename, "Compiled JSON file")->required();
  auto* schema =
      app.add_subcommand("schema", "Print the compiled program JSON Schema");
  schema->add_option("-o,--out", output_directory,
                     "Output schema file (default: stdout)");
  auto* rewrite =
      app.add_subcommand("rewrite", "Greedily rewrite a graph and print TEPL");
  rewrite->add_option("file", filename, "TEPL file/project or compiled JSON")
      ->required();
  rewrite->add_option("-o,--out", output_directory,
                      "Output TEPL file (default: stdout)");
  std::string graph_name;
  std::string rules_path;
  tepl::rewrite::Options rewrite_options;
  rewrite->add_option("--graph", graph_name,
                      "Graph to rewrite (required when ambiguous)");
  for (auto* command : {compile, rewrite})
    command->add_option("--rules", rules_path,
                        "Rule source file or directory (source input only)");
  rewrite->add_option("--rule", rewrite_options.rule,
                      "Apply only this rule (default: all loaded rules)");
  rewrite->add_option("--max-steps", rewrite_options.max_steps,
                      "Maximum committed rewrites");
  rewrite->add_option("--max-nodes", rewrite_options.max_nodes,
                      "Maximum graph node count");
  rewrite->add_flag("--trace", rewrite_options.trace,
                    "Report applied rules on stderr");
  std::string semantic_policy = "strict";
  std::string semantic_rule;
  check->add_option("--rule", semantic_rule,
                    "Select a semantic rule for compile-time proof");
  for (auto* command : {check, generate, compile, rewrite}) {
    command
        ->add_option("--semantic-policy", semantic_policy,
                     "Arithmetic proof policy (strict or algebraic)")
        ->check(CLI::IsMember({"strict", "algebraic"}));
  }
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& error) {
    return app.exit(error) == 0 ? 0 : 2;
  }

  // bazel run starts in the binary's runfiles directory. User paths belong to
  // the directory where bazel was invoked, including relative output paths.
  if (const auto* working_directory = std::getenv("BUILD_WORKING_DIRECTORY")) {
    if (*working_directory) {
      try {
        std::filesystem::current_path(working_directory);
      } catch (const std::filesystem::filesystem_error& error) {
        std::cerr << "cannot use invocation directory: " << error.what()
                  << '\n';
        return 2;
      }
    }
  }

  if (*init) {
    try {
      tepl::support::initializeProject(project_directory);
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      return 2;
    }
    std::cout << "Initialized TEPL project in " << project_directory << ".\n"
              << "Run tepl check \"" << project_directory
              << "\" to check the starter project.\n";
    return 0;
  }

  const auto execute = [&](const tepl::program::Program& executable) -> int {
    if (*validate) {
      std::cout << "Compiled program is valid.\n";
      return 0;
    }
    if (*compile) {
      tepl::rewrite::writeOutput(tepl::program::serialize(executable),
                                 output_directory);
      return 0;
    }
    if (*rewrite)
      return tepl::rewrite::runCommand(executable, graph_name, output_directory,
                                       rewrite_options);
    for (const auto& operation : executable.ir().operations) {
      if (operation.semantic_definition) {
        std::cerr << "axis semantics supports checking and greedy rewriting; "
                     "Rust/C++ code generation is not implemented\n";
        return 1;
      }
    }
    generation_options.target = target == "rust" ? tepl::codegen::Target::kRust
                                : target == "cpp"
                                    ? tepl::codegen::Target::kCpp
                                    : tepl::codegen::Target::kPython;
    write_options.target = generation_options.target;
    const auto generated =
        tepl::codegen::generate(executable, generation_options);
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
  };
  try {
    if (*schema) {
      tepl::rewrite::writeOutput(std::string(tepl::program::schema()),
                                 output_directory);
      return 0;
    }
    if (*validate || ((*rewrite || *generate) &&
                      std::filesystem::path(filename).extension() == ".json")) {
      if (!rules_path.empty())
        throw std::invalid_argument(
            "--rules requires TEPL source input; compiled JSON already "
            "contains rules");
      std::ifstream stream(filename, std::ios::binary);
      if (!stream) throw std::runtime_error("cannot open " + filename);
      std::string text{std::istreambuf_iterator<char>(stream), {}};
      return execute(tepl::program::load(text));
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  tepl::ParseResult result;
  try {
    if (!rules_path.empty()) {
      result = tepl::loadRewriteSources(filename, rules_path);
    } else if ((*generate || *check || *compile || *rewrite) &&
               std::filesystem::is_directory(filename)) {
      result = tepl::loadProject(filename);
      generation_options.graphs_root =
          (std::filesystem::absolute(filename) / "graphs").string();
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
  if (*check || *generate || *compile || *rewrite) {
    auto analyzed = tepl::core::analyze(
        *result.program,
        {semantic_policy == "algebraic" ? tepl::core::SemanticPolicy::kAlgebraic
                                        : tepl::core::SemanticPolicy::kStrict,
         false, semantic_rule});
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
      if (semantic_rule.empty())
        std::cout << tepl::core::formatProgram(*analyzed.program);
      return tepl::checkSemanticRules(*analyzed.program, semantic_rule,
                                      std::cout);
    }
    try {
      return execute(tepl::program::compile(*analyzed.program));
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
  if (print_tree) {
    std::cout << result.tree << '\n';
  } else if (print_ast) {
    std::cout << tepl::formatAst(*result.program);
  } else {
    std::cout << "Parsed " << result.rule_count << " rule(s)";
    if (!result.program->graphs.empty())
      std::cout << " and " << result.program->graphs.size() << " graph(s)";
    std::cout << ".\n";
  }
}
