#include "src/index_check.h"
namespace tepl {
int checkIndexRules(const core::Program& program, const std::string& name,
                    std::ostream& output) {
  bool selected = false;
  bool proved = true;
  for (const auto& rule : program.rules) {
    if (!rule.index_inference || (!name.empty() && name != rule.name)) continue;
    selected = true;
    if (!name.empty()) {
      output << "IndexRule " << rule.name << '\n';
      output << core::index_semantics::format(*rule.index_inference);
    }
    proved &= rule.index_inference->proof.status ==
              core::index_semantics::ProofStatus::Proved;
  }
  if (!name.empty() && !selected) {
    output << "index rule: no index-semantic rule named '" << name << "'\n";
    return 2;
  }
  return proved ? 0 : 1;
}
}  // namespace tepl
