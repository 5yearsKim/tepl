#include "src/legacy_semantics/codec.h"

#include <iomanip>
#include <sstream>

namespace tepl::core::semantics {
namespace {
void writeNumbers(std::ostream& out, const auto& xs) {
  out << xs.size() << ' ';
  for (auto x : xs) out << x << ' ';
}
}  // namespace
std::string encodePlan(const inference::SemanticPlan& p) {
  std::ostringstream out;
  out << "tepl-constraints-2 " << p.source_root << ' ' << p.rhs_root << ' '
      << p.source_end << ' ' << p.algebraic << ' ' << p.timeout_ms << ' '
      << p.nodes.size() << ' ';
  for (auto& n : p.nodes) {
    out << n.capture << ' ';
    writeNumbers(out, n.inputs);
    out << n.root << ' ' << n.shape << ' ';
    writeNumbers(out, n.assertions);
    out << n.fields.size() << ' ';
    for (auto& f : n.fields)
      out << f.known << ' ' << f.output << ' ' << f.bound << ' ';
    out << n.expressions.size() << ' ';
    for (auto& e : n.expressions) {
      out << std::quoted(e.op) << ' ';
      writeNumbers(out, e.args);
    }
  }
  const auto& c = p.constraints;
  out << c.available << ' ' << std::quoted(c.reason) << ' '
      << c.expressions.size() << ' ';
  for (auto& e : c.expressions) {
    out << std::quoted(e.op) << ' ';
    writeNumbers(out, e.args);
  }
  writeNumbers(out, c.domains);
  for (auto* terms : {&c.lhs, &c.rhs}) {
    out << terms->size() << ' ';
    for (auto& t : *terms) {
      out << t.coefficient << ' ';
      writeNumbers(out, t.reductions);
      writeNumbers(out, t.unused_reductions);
      out << t.reads.size() << ' ';
      for (auto& read : t.reads)
        out << read.capture << ' ' << read.indices << ' ';
    }
  }
  return out.str();
}
}  // namespace tepl::core::semantics
