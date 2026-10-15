#include <algorithm>

#include "src/rewrite/builtins/implementation.h"
#include "src/rewrite/evaluation_support.h"
namespace tepl::rewrite::detail {
Value listBuiltin(core::builtins::Builtin id, const std::vector<Value>& a,
                  Budget& budget) {
  using B = core::builtins::Builtin;
  switch (id) {
    case B::kLen: {
      return Integer(list(a.at(0)).size());
    }
    case B::kRange: {
      auto count = position(a.at(0));
      budget.size(count);
      List xs;
      for (std::size_t i = 0; i < count; ++i) xs.emplace_back(Integer(i));
      return xs;
    }
    case B::kConcat: {
      List xs;
      for (const auto& v : a) {
        const auto& part = list(v);
        budget.size(xs.size() + part.size());
        xs.insert(xs.end(), part.begin(), part.end());
      }
      return xs;
    }
    case B::kGather: {
      List xs;
      const auto& src = list(a.at(0));
      for (const auto& p : list(a.at(1))) {
        auto i = position(p);
        ensure(i < src.size(), "gather out of bounds");
        xs.push_back(src[i]);
      }
      return xs;
    }
    case B::kSlice: {
      const auto& xs = list(a.at(0));
      auto b = position(a.at(1)), e = position(a.at(2));
      ensure(b <= e && e <= xs.size(), "invalid slice");
      return List(xs.begin() + b, xs.begin() + e);
    }
    case B::kReplace: {
      auto xs = list(a.at(0));
      auto i = position(a.at(1));
      ensure(i < xs.size(), "replace out of bounds");
      xs[i] = a.at(2);
      return xs;
    }
    case B::kContains: {
      return contains(list(a.at(0)), a.at(1), budget);
    }
    case B::kAll:
    case B::kAny: {
      bool out = id == B::kAll;
      for (const auto& v : list(a.at(0)))
        if (boolean(v) != (id == B::kAll)) return id == B::kAny;
      return out;
    }
    case B::kExclude: {
      List out;
      for (const auto& x : list(a.at(0)))
        if (!contains(list(a.at(1)), x, budget)) out.push_back(x);
      return out;
    }
    case B::kIsDisjoint: {
      for (const auto& x : list(a.at(0)))
        if (contains(list(a.at(1)), x, budget)) return false;
      return true;
    }
    default:
      throw EvaluationError("invalid list builtin");
  }
}
}  // namespace tepl::rewrite::detail
