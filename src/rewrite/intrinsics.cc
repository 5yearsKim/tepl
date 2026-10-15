#include "src/rewrite/intrinsics.h"

#include <algorithm>

#include "src/core/builtins/axes.h"
#include "src/rewrite/builtins/implementation.h"
#include "src/rewrite/evaluation_support.h"
#include "src/rewrite/operators.h"
namespace tepl::rewrite {
Value evaluateIntrinsic(core::intrinsics::Intrinsic id,
                        const std::vector<Value>& a, Budget& budget) {
  using namespace detail;
  using I = core::intrinsics::Intrinsic;
  ensure(core::intrinsics::signature(id).acceptsArity(a.size()),
         "invalid intrinsic arity");
  if (id != I::kIndex) chargeArguments(a, budget);
  switch (id) {
    case I::kEmptyAxes: {
      return List{};
    }
    case I::kConcat: {
      return detail::listBuiltin(core::builtins::Builtin::kConcat, a, budget);
    }
    case I::kIndex:
      return evaluateIndex(a.at(0), a.at(1), budget);
    case I::kConditional: {
      return boolean(a.at(0)) ? a.at(1) : a.at(2);
    }
    case I::kAxisPartition: {
      const auto& xs = list(a.at(0));
      auto rank = integer(a.at(1));
      return valid(xs, rank, budget) && Integer(xs.size()) == rank;
    }
    case I::kInverse: {
      const auto& p = list(a.at(0));
      ensure(valid(p, p.size(), budget), "invalid permutation");
      List out(p.size());
      for (std::size_t i = 0; i < p.size(); ++i)
        out[position(p[i])] = Integer(i);
      return out;
    }
    case I::kOrderPositions: {
      const auto& p = list(a.at(0));
      List out;
      for (std::size_t i = 0; i < p.size(); ++i) out.emplace_back(Integer(i));
      std::stable_sort(
          out.begin(), out.end(), [&](const auto& x, const auto& y) {
            return integer(p[position(x)]) < integer(p[position(y)]);
          });
      return out;
    }
    case I::kOffsetPositions:
    case I::kOffsetExtents:
    case I::kScaleExtents: {
      auto xs = list(a.at(0));
      for (auto& x : xs)
        x = arithmetic(id == I::kScaleExtents ? "*" : "+", integer(x),
                       integer(a.at(1)));
      return xs;
    }
    case I::kExtentAdd:
    case I::kExtentSub:
    case I::kExtentMul:
    case I::kExtentFloorDiv:
    case I::kExtentLe: {
      const auto& x = list(a.at(0));
      const auto& y = list(a.at(1));
      ensure(x.size() == y.size(), "extent lengths differ");
      List out;
      bool le = true;
      for (std::size_t i = 0; i < x.size(); ++i) {
        le &= integer(x[i]) <= integer(y[i]);
        if (id == I::kExtentFloorDiv) {
          ensure(integer(y[i]) > 0, "invalid extent divisor");
          out.push_back(evaluateBuiltin(core::builtins::Builtin::kFloorDiv,
                                        {x[i], y[i]}, budget));
        } else if (id != I::kExtentLe)
          out.emplace_back(arithmetic(id == I::kExtentAdd   ? "+"
                                      : id == I::kExtentSub ? "-"
                                                            : "*",
                                      integer(x[i]), integer(y[i])));
      }
      return id == I::kExtentLe ? Value(le) : Value(out);
    }
    case I::kIsValidExtents:
    case I::kIsPositiveExtents: {
      for (const auto& x : list(a.at(0)))
        if (id == I::kIsValidExtents ? integer(x) < 0 : integer(x) <= 0)
          return false;
      return true;
    }
    case I::kDistinctAxes: {
      return distinct(list(a.at(0)), budget);
    }
    case I::kAxisExtents: {
      List out;
      for (const auto& x : list(a.at(0)))
        out.emplace_back(std::get<Axis>(x.data).extent);
      return out;
    }
    case I::kPositions: {
      const auto& src = list(a.at(0));
      const auto& wanted = list(a.at(1));
      ensure(distinct(src, budget) && distinct(wanted, budget),
             "noninjective positions");
      List out;
      for (const auto& v : wanted) {
        budget.charge(src.size());
        auto it = std::find(src.begin(), src.end(), v);
        ensure(it != src.end(), "absent axis");
        out.emplace_back(Integer(it - src.begin()));
      }
      return out;
    }
    case I::kMembershipPositions:
    case I::kMatchedPositions: {
      const auto& x = list(a.at(0));
      const auto& y = list(a.at(1));
      List out;
      for (std::size_t i = 0; i < x.size(); ++i) {
        budget.charge(y.size());
        auto it = std::find(y.begin(), y.end(), x[i]);
        if (it != y.end())
          out.emplace_back(
              Integer(id == I::kMembershipPositions ? i : it - y.begin()));
      }
      return out;
    }
    case I::kUnionAxes:
    case I::kUniqueAxes: {
      List out;
      for (const auto& v : a)
        for (const auto& x : list(v))
          if (!contains(out, x, budget)) out.push_back(x);
      return out;
    }
    case I::kOrderedComplement:
    case I::kDifferenceAxes:
      return detail::listBuiltin(core::builtins::Builtin::kExclude, a, budget);
    case I::kDisjoint:
      return detail::listBuiltin(core::builtins::Builtin::kIsDisjoint, a,
                                 budget);
    case I::kFilterAxes:
    case I::kIntersectionAxes:
    case I::kSameSet:
    case I::kSubset: {
      const auto& xs = list(a.at(0));
      const auto& ys = list(a.at(1));
      List out;
      for (const auto& x : xs) {
        bool present = contains(ys, x, budget);
        if (present) out.push_back(x);
      }
      if (id == I::kSubset) return out.size() == xs.size();
      if (id == I::kSameSet) {
        for (const auto& y : ys)
          if (!contains(xs, y, budget)) return false;
        return out.size() == xs.size();
      }
      return out;
    }
    case I::kReplaceAxes: {
      auto xs = list(a.at(0));
      const auto& from = list(a.at(1));
      const auto& to = list(a.at(2));
      ensure(from.size() == to.size() && distinct(from, budget) &&
                 distinct(to, budget),
             "invalid axis replacement");
      for (auto& x : xs) {
        budget.charge(from.size());
        auto it = std::find(from.begin(), from.end(), x);
        if (it != from.end()) x = to[it - from.begin()];
      }
      return xs;
    }
    case I::kPlaceAxes: {
      const auto& left = list(a.at(0));
      const auto& p = list(a.at(1));
      const auto& right = list(a.at(2));
      ensure(left.size() == p.size() &&
                 valid(p, left.size() + right.size(), budget),
             "invalid axis placement");
      budget.size(left.size() + right.size());
      List out(left.size() + right.size());
      std::vector<bool> occupied(out.size());
      for (std::size_t i = 0; i < left.size(); ++i) {
        auto at = position(p[i]);
        out[at] = left[i];
        occupied[at] = true;
      }
      std::size_t next = 0;
      for (std::size_t i = 0; i < out.size(); ++i)
        if (!occupied[i]) out[i] = right[next++];
      return out;
    }
  }
  throw EvaluationError("invalid intrinsic ID");
}
Value evaluateAxisCall(std::string_view name, const std::vector<Value>& args,
                       Budget& budget) {
  const auto* entry = core::builtins::resolveAxisBuiltin(name);
  detail::ensure(entry && entry->acceptsArity(args.size()),
                 "invalid axes builtin");
  return std::visit(
      [&](auto target) -> Value {
        using T = decltype(target);
        if constexpr (std::is_same_v<T, core::builtins::Builtin>)
          return evaluateBuiltin(target, args, budget);
        else if constexpr (std::is_same_v<T, core::intrinsics::Intrinsic>)
          return evaluateIntrinsic(target, args, budget);
        else {
          using Q = core::builtins::AxisQuery;
          if (target == Q::Rank)
            return evaluateBuiltin(core::builtins::Builtin::kLen, args, budget);
          if (target == Q::AllAxes) {
            // The operand is represented by its concrete shape at this
            // boundary.
            detail::chargeArguments(args, budget);
            return detail::listBuiltin(core::builtins::Builtin::kRange,
                                       {Integer(list(args.at(0)).size())},
                                       budget);
          }
          detail::chargeArguments(args, budget);
          return args.at(0);
        }
      },
      entry->target);
}
Value evaluateSymbolicCall(std::string_view name,
                           const std::vector<Value>& args, Budget& budget) {
  if (const auto* entry = core::intrinsics::resolve(name))
    return evaluateIntrinsic(entry->intrinsic, args, budget);
  if (const auto* entry = core::builtins::resolveBuiltin(name))
    return evaluateBuiltin(entry->builtin, args, budget);
  if (name.starts_with("dtype_")) {
    detail::ensure(
        args.empty() && core::resolveDType(name.substr(6)).has_value(),
        "invalid dtype constant");
    budget.tick();
    return std::string(name.substr(6));
  }
  return evaluateOperator(name, args, budget);
}
}  // namespace tepl::rewrite
