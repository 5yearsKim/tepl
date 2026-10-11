#include "src/core/semantics/ir.h"
namespace tepl::core::semantics {
std::string format(const Expr& e) {
  std::string out = e.op;
  if (!e.arguments.empty()) {
    out += '(';
    for (std::size_t i = 0; i < e.arguments.size(); ++i) {
      if (i) out += ',';
      out += format(*e.arguments[i]);
    }
    out += ')';
  }
  return out;
}
}  // namespace tepl::core::semantics
