#include "src/rewrite/cli.h"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "src/rewrite/printer.h"
namespace tepl::rewrite {
void writeOutput(const std::string& text, const std::string& output) {
  if (output.empty()) {
    std::cout << text;
    return;
  }
  std::ofstream stream(output, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot open output '" + output + "'");
  stream << text;
  stream.close();
  if (!stream) throw std::runtime_error("cannot write output '" + output + "'");
}
int runCommand(const executable::ExecutableProgram& p, const std::string& name,
               const std::string& output, const Options& options) {
  const core::ConcreteGraph* graph = nullptr;
  for (const auto& g : p.ir().graphs)
    if (name.empty() || g.name == name) {
      if (graph)
        throw std::invalid_argument(
            "select one graph with --graph; graph selection is ambiguous");
      graph = &g;
    }
  if (!graph) throw std::invalid_argument("no matching graph in program");
  auto result = run(p, *graph, options);
  auto text = printGraph(p.ir(), result.graph);
  writeOutput(text, output);
  for (const auto& step : result.trace)
    std::cerr << "Successfully applied rule '" << step.rule << "'.\n";
  std::cerr << "Rewrites: " << result.rewrite_count
            << "; stop: " << stopReasonName(result.stop_reason)
            << "; policy: " << p.policy() << '\n';
  if (!result.detail.empty()) std::cerr << result.detail << '\n';
  if (result.stop_reason == StopReason::FixedPoint)
    std::cerr << "Rewrite completed.";
  else
    std::cerr << "Rewrite stopped (" << stopReasonName(result.stop_reason)
              << ").";
  if (!output.empty())
    std::cerr << " Result saved to " << output << ".\n";
  else
    std::cerr << " Result written to stdout.\n";
  return result.stop_reason == StopReason::FixedPoint ? 0 : 3;
}
}  // namespace tepl::rewrite
