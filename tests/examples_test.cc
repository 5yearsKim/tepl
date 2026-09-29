#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/parse.h"

int main(int argc, char** argv) {
  using rules_cc::cc::runfiles::Runfiles;
  std::string error;
  std::unique_ptr<Runfiles> runfiles(
      Runfiles::CreateForTest(BAZEL_CURRENT_REPOSITORY, &error));
  if (!runfiles || argc != 6) {
    std::cerr << "Cannot load example runfiles: " << error << '\n';
    return 1;
  }

  const std::size_t expected_counts[] = {1, 1, 2, 2, 1};
  for (int i = 1; i < argc; ++i) {
    std::ifstream input(runfiles->Rlocation(argv[i]));
    if (!input) {
      std::cerr << "Cannot open example: " << argv[i] << '\n';
      return 1;
    }
    std::string source{std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>()};
    auto result = tepl::Parse(source);
    for (const auto& diagnostic : result.diagnostics) {
      std::cerr << argv[i] << ':' << diagnostic.line << ':' << diagnostic.column
                << ": " << diagnostic.message << '\n';
    }
    if (!result.ok() || result.rule_count != expected_counts[i - 1] ||
        result.tree.empty()) {
      std::cerr << "Unexpected parse result for " << argv[i] << '\n';
      return 1;
    }
  }
}
