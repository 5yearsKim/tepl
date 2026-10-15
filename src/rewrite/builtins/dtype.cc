#include <algorithm>

#include "src/rewrite/builtins/implementation.h"
#include "src/rewrite/evaluation_support.h"
namespace tepl::rewrite::detail {
Value dtypeBuiltin(core::builtins::Builtin id, const std::vector<Value>& a,
                   Budget& budget) {
  using B = core::builtins::Builtin;
  switch (id) {
    case B::kIsFloat:
    case B::kIsInteger:
    case B::kIsSignedInteger:
    case B::kIsUnsignedInteger:
    case B::kIsNumeric: {
      auto s = std::get<std::string>(a.at(0).data);
      ensure(core::resolveDType(s).has_value(), "unknown dtype");
      bool f = s[0] == 'f' || s == "bf16", i = s[0] == 'i', u = s[0] == 'u';
      return id == B::kIsFloat             ? f
             : id == B::kIsInteger         ? (i || u)
             : id == B::kIsSignedInteger   ? i
             : id == B::kIsUnsignedInteger ? u
                                           : (f || i || u);
    }
    default:
      throw EvaluationError("invalid dtype builtin");
  }
}
}  // namespace tepl::rewrite::detail
