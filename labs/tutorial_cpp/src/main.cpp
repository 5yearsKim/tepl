#include <cassert>
#include <eggc/extract.hpp>
#include <iostream>

#include "generated/generated.h"
namespace ir = tutorial_ir;
int main() {
  auto built = ir::graphs::program::graph_sample::build().value();
  auto rules = ir::rules::all().value();
  auto report = tepl_runtime::runner::run(built.graph, rules).value();
  assert(report.reason == eggc::StopReason::Saturated);
  assert(built.graph.find(built.root) ==
         built.graph.find(built.bindings.at("X")));
  const auto& facts = built.graph.analysis_data(built.root);
  assert((facts == tepl_runtime::TensorFacts::known({2, 3}, ir::DType::F32)));
  auto [cost, best] =
      eggc::Extractor<ir::OpNode, ir::TensorAnalysis>(built.graph)
          .find_best(built.root);
  assert(cost == 1);
  std::cout << "Saturated; transpose composition reduced to X (cost " << cost
            << ").\n";
}
