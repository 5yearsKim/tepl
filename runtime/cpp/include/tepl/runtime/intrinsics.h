#pragma once
#include "builtins.h"
namespace tepl_runtime::intrinsics {
inline Value call(std::string_view name, const std::vector<Value>& a,
                  Budget& b) {
  using namespace builtins;
  std::optional<std::size_t> arity = 2;
  if (name == "empty_axes") arity = 0;
  if (name == "inverse" || name == "order_positions" ||
      name == "is_valid_extents" || name == "is_positive_extents" ||
      name == "distinct_axes" || name == "axis_extents" ||
      name == "unique_axes")
    arity = 1;
  if (name == "replace_axes" || name == "place_axes" || name == "if") arity = 3;
  if (name == "concat" || name == "union_axes") arity.reset();
  require(!arity || *arity == a.size(), "invalid intrinsic arity");
  if (name != "index") b.arguments(a);
  if (name == "empty_axes") return Value(Value::List{});
  if (name == "concat") return concat(a, b);
  if (name == "index") return index(a[0], a[1], b);
  if (name == "if") return a[0].boolean() ? a[1] : a[2];
  if (name == "axis_partition")
    return Value(valid(a[0].list(), a[1].integer(), b) &&
                 Integer(a[0].list().size()) == a[1].integer());
  if (name == "inverse") {
    auto& xs = a[0].list();
    require(valid(xs, Integer(xs.size()), b), "invalid permutation");
    Value::List out(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i)
      out[xs[i].position()] = Value(Integer(i));
    return Value(out);
  }
  if (name == "order_positions") {
    auto& xs = a[0].list();
    std::vector<std::size_t> order(xs.size());
    std::iota(order.begin(), order.end(), 0);
    for (const auto& x : xs) x.integer();
    std::stable_sort(order.begin(), order.end(), [&](auto i, auto j) {
      return xs[i].integer() < xs[j].integer();
    });
    Value::List out;
    for (auto i : order) out.emplace_back(Integer(i));
    return Value(out);
  }
  if (name == "offset_positions" || name == "offset_extents" ||
      name == "scale_extents") {
    Value::List out;
    for (const auto& x : a[0].list())
      out.emplace_back(arithmetic(name == "scale_extents" ? "*" : "+",
                                  x.integer(), a[1].integer()));
    return Value(out);
  }
  if (name == "extent_add" || name == "extent_sub" || name == "extent_mul" ||
      name == "extent_floor_div" || name == "extent_le") {
    auto& xs = a[0].list();
    auto& ys = a[1].list();
    require(xs.size() == ys.size(), "extent lengths differ");
    Value::List out;
    bool le = true;
    for (std::size_t i = 0; i < xs.size(); ++i) {
      le = le && (xs[i].integer() <= ys[i].integer());
      if (name == "extent_floor_div")
        out.push_back(builtins::call("floor_div", {xs[i], ys[i]}, b));
      else if (name != "extent_le")
        out.emplace_back(arithmetic(name == "extent_add"   ? "+"
                                    : name == "extent_sub" ? "-"
                                                           : "*",
                                    xs[i].integer(), ys[i].integer()));
    }
    return name == "extent_le" ? Value(le) : Value(out);
  }
  if (name == "is_valid_extents" || name == "is_positive_extents") {
    for (const auto& x : a[0].list())
      if (name == "is_valid_extents" ? x.integer() < 0 : x.integer() <= 0)
        return Value(false);
    return Value(true);
  }
  if (name == "distinct_axes") return Value(distinct(a[0].list(), b));
  if (name == "axis_extents") {
    Value::List out;
    for (const auto& x : a[0].list())
      out.emplace_back(x.get<Axis>("expected axis").extent);
    return Value(out);
  }
  if (name == "positions") {
    auto& xs = a[0].list();
    auto& ys = a[1].list();
    require(distinct(xs, b) && distinct(ys, b), "noninjective positions");
    Value::List out;
    for (const auto& x : ys) {
      b.charge(xs.size());
      auto i = std::find(xs.begin(), xs.end(), x);
      require(i != xs.end(), "absent axis");
      out.emplace_back(Integer(i - xs.begin()));
    }
    return Value(out);
  }
  if (name == "membership_positions" || name == "matched_positions") {
    auto& xs = a[0].list();
    auto& ys = a[1].list();
    Value::List out;
    for (std::size_t i = 0; i < xs.size(); ++i) {
      b.charge(ys.size());
      auto j = std::find(ys.begin(), ys.end(), xs[i]);
      if (j != ys.end())
        out.emplace_back(Integer(
            name == "membership_positions" ? i : std::size_t(j - ys.begin())));
    }
    return Value(out);
  }
  if (name == "union_axes" || name == "unique_axes") {
    Value::List out;
    for (const auto& v : a)
      for (const auto& x : v.list())
        if (!contains(out, x, b)) out.push_back(x);
    return Value(out);
  }
  if (name == "ordered_complement" || name == "difference_axes")
    return exclude(a, b);
  if (name == "disjoint") return disjoint(a, b);
  if (name == "filter_axes" || name == "intersection_axes" ||
      name == "same_set" || name == "subset") {
    auto& xs = a[0].list();
    auto& ys = a[1].list();
    Value::List out;
    for (const auto& x : xs)
      if (contains(ys, x, b)) out.push_back(x);
    if (name == "subset") return Value(out.size() == xs.size());
    if (name == "same_set") {
      for (const auto& y : ys)
        if (!contains(xs, y, b)) return Value(false);
      return Value(out.size() == xs.size());
    }
    return Value(out);
  }
  if (name == "replace_axes") {
    auto xs = a[0].list();
    auto& from = a[1].list();
    auto& to = a[2].list();
    require(from.size() == to.size() && distinct(from, b) && distinct(to, b),
            "invalid axis replacement");
    for (auto& x : xs) {
      b.charge(from.size());
      auto i = std::find(from.begin(), from.end(), x);
      if (i != from.end()) x = to[i - from.begin()];
    }
    return Value(xs);
  }
  if (name == "place_axes") {
    auto& left = a[0].list();
    auto& p = a[1].list();
    auto& right = a[2].list();
    require(
        right.size() <= std::numeric_limits<std::size_t>::max() - left.size(),
        "list size overflow");
    auto n = left.size() + right.size();
    require(left.size() == p.size() && valid(p, Integer(n), b),
            "invalid axis placement");
    b.size(n);
    std::vector<std::optional<Value>> slots(n);
    for (std::size_t i = 0; i < left.size(); ++i)
      slots[p[i].position()] = left[i];
    Value::List out;
    std::size_t at = 0;
    for (auto& x : slots) out.push_back(x ? *x : right[at++]);
    return Value(out);
  }
  throw EvalError(ErrorKind::Invalid,
                  "unknown intrinsic: " + std::string(name));
}
}  // namespace tepl_runtime::intrinsics
