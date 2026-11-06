#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include "CLI/CLI.hpp"
#include "src/cli/semantic_report.h"
#include "src/cli/core_emission.h"
#include "src/codegen/generate.h"
#include "src/codegen/write.h"
#include "src/core/pipeline.h"
#include "src/core/resolution/loading.h"
#include "src/core/semantics/report.h"
#include "src/core/syntax/parse.h"
#include "src/core/syntax/print.h"
#include "src/executable/program.h"
#include "src/rewrite/cli.h"
#include "src/support/init.h"
#include "src/support/project_config.h"

int main(int argc, char** argv) {
  CLI::App app{"Compile, rewrite, and generate code for TEPL projects"};
  app.set_version_flag("--version", tepl::support::kTeplVersion);
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
  auto* core_emit = app.add_subcommand("core-emit", "Emit core metadata DAG without verifying or accepting a rewrite");
  core_emit->add_option("file", filename, "TEPL source or project")->required();
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
      "Format generated code with rustfmt or clang-format (default: on)");
  generate->add_option("--cpp-namespace", generation_options.cpp_namespace,
                       "Namespace for generated C++ declarations");
  generate->add_option("--rules-root", generation_options.rules_root,
                       "Root for generated rule module paths");
  generate->add_option("--graphs-root", generation_options.graphs_root,
                       "Root for generated graph module paths");
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
  for (auto* command : {compile, rewrite, generate})
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
  bool core_backend=false;
  for(auto* command:{check,compile,rewrite,generate}) {
    command->add_flag("--core",core_backend,"Use the metadata DAG constructor and supported schemas");
  }
  std::string semantic_policy;
  std::string semantic_rule;
  core_emit->add_option("--rule", semantic_rule, "Select a core rule");
  check->add_option("--rule", semantic_rule,
                    "Select a semantic rule for compile-time proof");
  for (auto* command : {check, generate, compile, rewrite, core_emit}) {
    command
        ->add_option("--semantic-policy", semantic_policy,
                     "Arithmetic proof policy (strict or algebraic)")
        ->check(CLI::IsMember({"strict", "algebraic"}))
        ->default_str("strict (or project configuration)");
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

  const auto execute =
      [&](const tepl::executable::ExecutableProgram& executable) -> int {
    try {
      if (*validate) {
        std::cout << "Compiled program is valid.\n";
        return 0;
      }
      if (*compile) {
        tepl::rewrite::writeOutput(tepl::executable::serialize(executable),
                                   output_directory);
        return 0;
      }
      if (*rewrite)
        return tepl::rewrite::runCommand(executable, graph_name,
                                         output_directory, rewrite_options);
      generation_options.target =
          target == "rust"  ? tepl::codegen::Target::kRust
          : target == "cpp" ? tepl::codegen::Target::kCpp
                            : tepl::codegen::Target::kPython;
      auto generated = tepl::codegen::generate(executable, generation_options);
      for (const auto& error : generated.diagnostics) {
        const auto& at = error.origin.definition;
        if (!at.source_name.empty())
          std::cerr << at.source_name << ':' << at.span.begin.line << ':'
                    << at.span.begin.column << ": ";
        std::cerr << error.message << '\n';
      }
      if (!generated.ok()) return 1;
      auto written = tepl::codegen::synchronize(
          generated.files, output_directory, write_options);
      if (write_options.check && !written.current()) {
        for (const auto& path : written.differences)
          std::cerr << "generated output differs: " << path << '\n';
        return 1;
      }
      return 0;
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  };
  try {
    if (*schema) {
      tepl::rewrite::writeOutput(std::string(tepl::executable::schema()),
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
      return execute(tepl::executable::load(text));
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  tepl::LoadResult result;
  try {
    if (!rules_path.empty()) {
      result = tepl::loadRewriteSources(filename, rules_path);
    } else if ((*generate || *check || *compile || *rewrite || *core_emit) &&
               std::filesystem::is_directory(filename)) {
      result = tepl::loadProject(filename);
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
      auto parsed = tepl::parse(source, filename);
      if (parsed.ok())
        result = tepl::loadImports(*parsed.program);
      else
        result.diagnostics = std::move(parsed.diagnostics);
      result.tree = std::move(parsed.tree);
    }
  } catch (const std::filesystem::filesystem_error& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
  for (const auto& diagnostic : result.diagnostics) {
    std::cerr << (diagnostic.origin.definition.source_name.empty()
                      ? filename
                      : diagnostic.origin.definition.source_name)
              << ':' << diagnostic.origin.definition.span.begin.line << ':'
              << diagnostic.origin.definition.span.begin.column << ": "
              << diagnostic.message << '\n';
  }
  if (!result.ok()) {
    return 1;
  }
  if (*check || *generate || *compile || *rewrite || *core_emit) {
    if (semantic_policy.empty())
      semantic_policy = result.config ? result.config->semantic_policy : "strict";
    const auto policy = semantic_policy == "algebraic"
                            ? tepl::core::SemanticPolicy::kAlgebraic
                            : tepl::core::SemanticPolicy::kStrict;
    auto diagnostics = [](const std::vector<tepl::Diagnostic>& errors) {
      for (const auto& error : errors) {
        const auto location = [](const tepl::SourceLocation& at) {
          std::cerr << at.source_name << ':' << at.span.begin.line << ':'
                    << at.span.begin.column;
        };
        location(error.origin.definition);
        std::cerr << ": " << error.message << '\n';
        for (const auto& at : error.origin.expansions) {
          location(at);
          std::cerr << ": note: instantiated here\n";
        }
        for (const auto& at : error.related)
          if (!at.source_name.empty()) {
            location(at);
            std::cerr << ": note: related declaration or expression\n";
          }
      }
    };
    if (*core_emit) {
      tepl::core::InspectionOptions options{policy, true};
      options.defer_semantic_proofs = true;
      auto inspected = tepl::core::inspect(*result.program, options);
      diagnostics(inspected.diagnostics);
      if (!inspected.ok()) return 1;
      try { std::cout << tepl::coreEmission(*inspected.program, semantic_rule); }
      catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
      return 0;
    }
    if (*check) {
      tepl::core::InspectionOptions options{policy,true,semantic_rule};
      options.core_backend=core_backend;
      auto inspected = tepl::core::inspect(*result.program,options);
      diagnostics(inspected.diagnostics);
      if (!inspected.ok()) return 1;
      if (semantic_rule.empty())
        std::cout << tepl::core::formatProgram(*inspected.program);
      return tepl::checkSemanticRules(*inspected.program, semantic_rule,
                                      std::cout);
    }
    if(core_backend && *generate){std::cerr<<"core DAG export currently supports portable JSON and native rewrite execution\n";return 1;}
    auto compiled = tepl::core::compile(*result.program, {policy,core_backend});
    diagnostics(compiled.diagnostics);
    if (!compiled.ok()) return 1;
    return execute(*compiled.program);
  }

  if (print_tree) {
    std::cout << result.tree << '\n';
  } else if (print_ast) {
    std::cout << tepl::formatAst(*result.program);
  } else {
    std::cout << "Parsed " << result.rule_count << " rule(s)";
    if (!result.program->declarations.graphs.empty())
      std::cout << " and " << result.program->declarations.graphs.size()
                << " graph(s)";
    std::cout << ".\n";
  }
}
