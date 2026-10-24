#include "src/core/attr_synth/ordered/concrete.h"

namespace tepl::core::attr_synth::ordered {
using namespace semantics;
ConcreteResult solve(const Plan& plan, const Metadata& metadata,
                     std::size_t max_work) {
  auto compiled = compile(plan);
  if (!compiled.program) {
    ConcreteResult out;
    out.reason = compiled.reason;
    return out;
  }
  return inference::evaluate(*compiled.program, metadata, {max_work});
}
bool check(const Plan& plan, const Metadata& metadata,
           const ConcreteResult& result) {
  auto compiled = compile(plan);
  return compiled.program &&
         inference::check(*compiled.program, metadata, result);
}
}  // namespace tepl::core::attr_synth::ordered
