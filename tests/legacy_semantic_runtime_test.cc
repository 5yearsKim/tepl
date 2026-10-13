#include <iostream>
#include <stdexcept>

#include "src/legacy_semantics/solver.h"

int main() {
  using namespace tepl::core;
  auto expect = [](bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
  };
  try {
    expect(semantics::encodePlan(inference::SemanticPlan{}) ==
               "tepl-constraints-2 0 0 0 0 1000 0 0 \"\" 0 0 0 0 ",
           "legacy serialization format changed");
    inference::SemanticPlan identity;
    identity.captures.push_back(CaptureId{0});
    inference::SemanticNode input;
    input.capture = 0;
    identity.nodes.push_back(input);
    identity.source_end = 1;
    auto request = semantics::encodePlan(identity) + "1 2 2 3 0 0 ";
    expect(semantics::solveRequest(request).starts_with("0 "),
           "legacy generated-runtime request no longer decodes or solves");
    std::cout << "legacy runtime compatibility passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
