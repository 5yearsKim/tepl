#include "src/cli/semantic_report.h"
namespace tepl {
int checkSemanticRules(const core::semantics::InspectionData& program,
                       const std::string& name, std::ostream& output) {
  bool selected = false;
  bool proved = true;
  for (const auto& rule : program.rules) {
    if (!rule.semantic_plan || (!name.empty() && name != rule.name)) continue;
    selected = true;
    if (!name.empty()) {
      output << "SemanticRule " << rule.name << '\n';
      output << core::semantics::format(*rule.semantic_plan);
    }
    const auto& proof = rule.semantic_plan->proof;
    if (proof.status != core::semantics::ProofStatus::Proved) {
      output << "error: cannot verify rule '" << rule.name
             << "': " << proof.reason
             << "\nAdd an explicit where condition that establishes the "
                "unresolved requirement, or change the rewrite.\n";
      proved = false;
    }
  }
  if (!name.empty() && !selected) {
    output << "semantic rule: no semantic rule named '" << name << "'\n";
    return 2;
  }
  return proved ? 0 : 1;
}
}  // namespace tepl
