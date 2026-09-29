#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>

#include "src/parse.h"

int main(int argc, char** argv) {
  if ((argc != 3 && argc != 4) || std::string_view(argv[1]) != "parse" ||
      (argc == 4 && std::string_view(argv[3]) != "--tree")) {
    std::cerr << "usage: tepl parse <file> [--tree]\n";
    return 2;
  }

  std::ifstream input(argv[2], std::ios::binary);
  if (!input) {
    std::cerr << argv[2] << ": cannot open file\n";
    return 2;
  }
  std::string source{std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>()};
  if (input.bad()) {
    std::cerr << argv[2] << ": cannot read file\n";
    return 2;
  }

  auto result = tepl::Parse(source);
  for (const auto& diagnostic : result.diagnostics) {
    std::cerr << argv[2] << ':' << diagnostic.line << ':' << diagnostic.column
              << ": " << diagnostic.message << '\n';
  }
  if (!result.ok()) {
    return 1;
  }
  if (argc == 4) {
    std::cout << result.tree << '\n';
  } else {
    std::cout << "Parsed " << result.rule_count << " rule(s).\n";
  }
}
