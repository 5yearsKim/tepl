#include <algorithm>

#include "src/rewrite/builtins/implementation.h"
#include "src/rewrite/evaluation_support.h"
namespace tepl::rewrite::detail {
Value integerBuiltin(core::builtins::Builtin id, const std::vector<Value>& a,
                     Budget& budget) {
  using B = core::builtins::Builtin;
  switch (id) {
    case B::kSum:
    case B::kProduct: {
      if (id == B::kProduct)
        for (const auto& v : list(a.at(0)))
          if (integer(v) == 0) return Integer(0);
      Integer out = id == B::kSum ? 0 : 1;
      for (const auto& v : list(a.at(0)))
        out = arithmetic(id == B::kSum ? "+" : "*", out, integer(v));
      return out;
    }
    case B::kMin: {
      return std::min(integer(a.at(0)), integer(a.at(1)));
    }
    case B::kMax: {
      return std::max(integer(a.at(0)), integer(a.at(1)));
    }
    case B::kFloorDiv:
    case B::kCeilDiv: {
      auto x = integer(a.at(0)), y = integer(a.at(1));
      ensure(y > 0, "rounded division requires positive divisor");
      auto q = arithmetic("/", x, y), r = arithmetic("%", x, y);
      if (r && ((r < 0) != (y < 0)) && id == B::kFloorDiv)
        q = arithmetic("-", q, 1);
      if (r && ((r < 0) == (y < 0)) && id == B::kCeilDiv)
        q = arithmetic("+", q, 1);
      return q;
    }
    default:
      throw EvaluationError("invalid integer builtin");
  }
}
}  // namespace tepl::rewrite::detail
