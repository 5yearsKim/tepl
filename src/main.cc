#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include "CLI/CLI.hpp"
#include "src/parse.h"

int main(int argc, char** argv) {
  CLI::App app{"Parse TEPL tensor rewrite rules"};
  app.require_subcommand(1);
  std::string filename;
  bool print_tree = false;
  auto* parse = app.add_subcommand("parse", "Validate a TEPL file");
  parse->add_option("file", filename, "TEPL input file")->required();
  parse->add_flag("--tree", print_tree, "Print the ANTLR parse tree");
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

  auto result = tepl::Parse(source);
  for (const auto& diagnostic : result.diagnostics) {
    std::cerr << filename << ':' << diagnostic.line << ':' << diagnostic.column
              << ": " << diagnostic.message << '\n';
  }
  if (!result.ok()) {
    return 1;
  }
  if (print_tree) {
    std::cout << result.tree << '\n';
  } else {
    std::cout << "Parsed " << result.rule_count << " rule(s).\n";
  }
}
