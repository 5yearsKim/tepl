#include "src/rewrite/builtins/implementation.h"
#include "src/rewrite/evaluation_support.h"
namespace tepl::rewrite {
Value evaluateBuiltin(core::builtins::Builtin id,
                      const std::vector<Value>& args, Budget& budget) {
  const auto& entry = core::builtins::signature(id);
  detail::ensure(entry.acceptsArity(args.size()), "invalid builtin arity");
  detail::chargeArguments(args, budget);
  using F = core::builtins::Feature;
  switch (entry.feature) {
    case F::kList:
      return detail::listBuiltin(id, args, budget);
    case F::kInteger:
      return detail::integerBuiltin(id, args, budget);
    case F::kShape:
      return detail::shapeBuiltin(id, args, budget);
    case F::kDType:
      return detail::dtypeBuiltin(id, args, budget);
  }
  throw EvaluationError("invalid builtin feature");
}
}  // namespace tepl::rewrite
