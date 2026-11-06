#include "src/cli/core_emission.h"

#include <nlohmann/json.hpp>

#include "src/core/attr_synth/ordered/compile.h"
namespace tepl {
std::string coreEmission(const core::semantics::InspectionData& p,
                         const std::string& name) {
  using J = nlohmann::json;
  J rules = J::array();
  const char* names[] = {"input",     "output",     "empty",    "union",
                         "intersect", "difference", "filter",   "concat",
                         "positions", "equal",      "same_set", "distinct",
                         "subset"};
  auto instructions = [](const auto& code) {
    J out = J::array();
    for (const auto& i : code)
      out.push_back(
          {{"kind", static_cast<int>(i.kind)},
           {"node", i.node},
           {"capture", i.capture ? J(*i.capture) : J(nullptr)},
           {"descriptor", i.descriptor ? J(*i.descriptor) : J(nullptr)},
           {"inputs", i.inputs},
           {"captures", i.captures},
           {"fields", i.fields}});
    return out;
  };
  for (const auto& r : p.rules) {
    if (!name.empty() && r.name != name) continue;
    if (!r.semantic_plan)
      throw std::invalid_argument("rule has no semantic plan");
    auto compiled = core::attr_synth::ordered::compile(*r.semantic_plan);
    if (!compiled.emission)
      throw std::invalid_argument(r.name + ": " + compiled.reason);
    J blocks = J::array();
    for (const auto& b : compiled.emission->blocks) {
      J nodes = J::array();
      for (const auto& e : b.expressions)
        nodes.push_back({{"op", names[static_cast<int>(e.op)]},
                         {"args", e.args},
                         {"capture", e.capture}});
      J attrs = J::array();
      for (const auto& [descriptor, fields] : b.attributes)
        attrs.push_back({{"descriptor", descriptor}, {"fields", fields}});
      blocks.push_back({{"source", instructions(b.source.source)},
                        {"replacement", instructions(b.source.replacement)},
                        {"expressions", nodes},
                        {"checks", b.checks},
                        {"attributes", attrs}});
    }
    rules.push_back({{"name", r.name}, {"blocks", blocks}});
  }
  if (rules.empty()) throw std::invalid_argument("no selected core rule");
  return J{{"format", "tepl.core-emission.v1"},
           {"accepted_rewrite", false},
           {"guard_verification", "not_attempted"},
           {"minimum_extent", 2},
           {"rules", rules}}
             .dump(2) +
         "\n";
}
}  // namespace tepl
