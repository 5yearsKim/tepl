#include <algorithm>

#include "src/rewrite/builtins/implementation.h"
#include "src/rewrite/evaluation_support.h"
namespace tepl::rewrite::detail {
Value shapeBuiltin(core::builtins::Builtin id, const std::vector<Value>& a,
                   Budget& budget) {
  using B = core::builtins::Builtin;
  switch (id) {
    case B::kIsValidAxisList: {
      return valid(list(a.at(0)), integer(a.at(1)), budget);
    }
    case B::kBroadcastShape: {
      const auto& x = list(a.at(0));
      const auto& y = list(a.at(1));
      List out(std::max(x.size(), y.size()));
      for (std::size_t i = 0; i < out.size(); ++i) {
        auto l = i < x.size() ? integer(x[x.size() - 1 - i]) : 1,
             r = i < y.size() ? integer(y[y.size() - 1 - i]) : 1;
        ensure(l >= 0 && r >= 0, "negative extent");
        ensure(l == r || l == 1 || r == 1, "incompatible broadcast");
        out[out.size() - 1 - i] = l == 1 ? r : l;
      }
      return out;
    }
    default:
      throw EvaluationError("invalid shape builtin");
  }
}
}  // namespace tepl::rewrite::detail
