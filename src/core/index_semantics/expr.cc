#include "src/core/index_semantics/expr.h"

#include <algorithm>
#include <mutex>

namespace tepl::core::index_semantics {
namespace {
std::string intern(const Symbol& s) {
  std::string signature = std::to_string(static_cast<int>(s.kind)) + ":" +
                          std::to_string(static_cast<int>(s.sort)) + ":";
  if (s.kind == Symbol::Kind::Field || s.kind == Symbol::Kind::Hole ||
      s.kind == Symbol::Kind::Shape || s.kind == Symbol::Kind::DType ||
      s.kind == Symbol::Kind::AxisGroup)
    signature += std::to_string(s.owner) + ":" + std::to_string(s.index);
  else
    signature += s.name;
  if (s.kind != Symbol::Kind::AxisGroup)
    for (const auto& a : s.args) signature += "{" + key(a) + "}";
  static std::mutex mutex;
  static std::map<std::string, std::size_t> identities;
  std::lock_guard<std::mutex> lock(mutex);
  auto [it, added] =
      identities.emplace(std::move(signature), identities.size() + 1);
  return "id" + std::to_string(it->second);
}
}  // namespace
S make(Symbol::Kind k, std::string name, std::size_t owner, std::size_t index,
       std::vector<S> args) {
  if (k == Symbol::Kind::AxisGroup && args.size() == 1 &&
      args[0]->kind == Symbol::Kind::List && args[0]->args.empty())
    return call("empty_axes", {});
  Symbol::Sort sort = Symbol::Sort::Positions;
  if (k == Symbol::Kind::Integer) sort = Symbol::Sort::Integer;
  if (k == Symbol::Kind::Boolean) sort = Symbol::Sort::Boolean;
  if (k == Symbol::Kind::Shape) sort = Symbol::Sort::Extents;
  if (k == Symbol::Kind::DType) sort = Symbol::Sort::DType;
  if (k == Symbol::Kind::AxisGroup) sort = Symbol::Sort::Axes;
  if (k == Symbol::Kind::Call) {
    if (name == "len" || name == "+" || name == "-" || name == "*" ||
        name == "index")
      sort = Symbol::Sort::Integer;
    else if (name == "==" || name == "!=" || name == "&&" || name == "||" ||
             name == "!" || name == "<" || name == "<=" || name == ">" ||
             name == ">=" || name == "axis_partition" ||
             name == "is_valid_axis_list" || name == "same_set" ||
             name == "disjoint" || name == "subset" || name.starts_with("is_"))
      sort = Symbol::Sort::Boolean;
    else if (name == "axis_extents")
      sort = Symbol::Sort::Extents;
    else if (name == "distinct_axes")
      sort = Symbol::Sort::Boolean;
    else if (name == "concat" || name == "gather" ||
             name == "ordered_complement" || name == "filter_axes" ||
             name == "difference_axes" || name == "union_axes" ||
             name == "intersection_axes" || name == "replace_axes" ||
             name == "unique_axes")
      sort = args.empty() ? Symbol::Sort::Axes : args[0]->sort;
    else if (name == "empty_axes")
      sort = Symbol::Sort::Axes;
    else if (name.starts_with("dtype_"))
      sort = Symbol::Sort::DType;
    else if (name == "if")
      sort = args.at(1)->sort;
  }
  Symbol node{k, std::move(name), owner, index, std::move(args), sort};
  node.structural_key = intern(node);
  return std::make_shared<Symbol>(std::move(node));
}
S integer(std::size_t n) {
  return make(Symbol::Kind::Integer, std::to_string(n));
}
S boolean(bool b) { return make(Symbol::Kind::Boolean, b ? "true" : "false"); }
std::string key(const S& s) {
  return s->structural_key.empty() ? intern(*s) : s->structural_key;
}
bool equal(const S& a, const S& b) { return key(a) == key(b); }
bool callIs(const S& s, const std::string& name) {
  return s->kind == Symbol::Kind::Call && s->name == name;
}
bool containsHole(const S& s) {
  if (s->kind == Symbol::Kind::Hole) return true;
  for (const auto& a : s->args)
    if (containsHole(a)) return true;
  return false;
}
S call(std::string name, std::vector<S> args) {
  auto empty = [](const S& s) {
    return (s->kind == Symbol::Kind::List && s->args.empty()) ||
           callIs(s, "empty_axes");
  };
  if (name == "len" && callIs(args[0], "concat")) {
    auto n = integer(0);
    for (const auto& a : args[0]->args) n = call("+", {n, call("len", {a})});
    return n;
  }
  if (name == "unique_axes" && args[0]->kind == Symbol::Kind::AxisGroup)
    return args[0];
  if (name == "unique_axes" && callIs(args[0], "concat")) {
    std::vector<S> xs;
    for (const auto& a : args[0]->args)
      if (std::none_of(xs.begin(), xs.end(),
                       [&](const S& x) { return equal(x, a); }))
        xs.push_back(a);
    if (xs.size() != args[0]->args.size())
      return call("unique_axes", {call("concat", std::move(xs))});
  }
  if (name == "+" && args[0]->kind == Symbol::Kind::Integer &&
      args[0]->name == "0")
    return args[1];
  if (name == "+" && args[1]->kind == Symbol::Kind::Integer &&
      args[1]->name == "0")
    return args[0];
  if (name == "+" && callIs(args[1], "-") && equal(args[0], args[1]->args[1]))
    return args[1]->args[0];
  if (name == "+" && callIs(args[0], "-") && equal(args[1], args[0]->args[1]))
    return args[0]->args[0];
  if (name == "-" && equal(args[0], args[1])) return integer(0);
  if (name == "len" && callIs(args[0], "offset_positions"))
    return call("len", {args[0]->args[0]});
  if (name == "offset_positions" && args[1]->kind == Symbol::Kind::Integer &&
      args[1]->name == "0")
    return args[0];
  if (name == "inverse" && callIs(args[0], "range")) return args[0];
  if (name == "len" && args[0]->kind == Symbol::Kind::AxisGroup)
    return call("len", {args[0]->args[0]});
  if (name == "len" && callIs(args[0], "inverse"))
    return call("len", {args[0]->args[0]});
  if (name == "len" && callIs(args[0], "empty_axes")) return integer(0);
  if ((name == "filter_axes" || name == "intersection_axes") &&
      equal(args[0], args[1]))
    return args[0];
  if ((name == "difference_axes") && equal(args[0], args[1]))
    return call("empty_axes", {});
  if ((name == "filter_axes" || name == "intersection_axes") &&
      (empty(args[0]) || empty(args[1])))
    return call("empty_axes", {});
  if (name == "difference_axes" && (empty(args[0]) || empty(args[1])))
    return args[0];
  if ((name == "same_set" || name == "subset") && equal(args[0], args[1]))
    return boolean(true);
  if (name == "disjoint" && (empty(args[0]) || empty(args[1])))
    return boolean(true);
  if (name == "union_axes") {
    std::vector<S> out;
    for (const auto& a : args) {
      if (empty(a)) continue;
      auto parts = callIs(a, "union_axes") ? a->args : std::vector<S>{a};
      for (const auto& part : parts)
        if (std::none_of(out.begin(), out.end(),
                         [&](const S& x) { return equal(x, part); }))
          out.push_back(part);
    }
    if (out.empty()) return call("empty_axes", {});
    if (out.size() == 1) return out[0];
    std::sort(out.begin(), out.end(),
              [](const S& a, const S& b) { return key(a) < key(b); });
    args = std::move(out);
  }
  if ((name == "is_valid_axis_list" || name == "axis_partition") &&
      callIs(args[0], "range") && equal(args[0]->args[0], args[1]))
    return boolean(true);
  if (name == "concat") {
    std::vector<S> flat;
    for (const auto& a : args) {
      if (callIs(a, "concat"))
        flat.insert(flat.end(), a->args.begin(), a->args.end());
      else if (!empty(a))
        flat.push_back(a);
    }
    if (flat.empty())
      return !args.empty() && args[0]->sort == Symbol::Sort::Axes
                 ? call("empty_axes", {})
                 : make(Symbol::Kind::List);
    if (flat.size() == 1) return flat[0];
    args = std::move(flat);
  }
  if (name == "len" && args[0]->kind == Symbol::Kind::List)
    return integer(args[0]->args.size());
  if (name == "len" && callIs(args[0], "range")) return args[0]->args[0];
  if (name == "len" && callIs(args[0], "gather"))
    return call("len", {args[0]->args[1]});
  if (name == "gather") {
    const auto& s = args[0];
    const auto& p = args[1];
    if (callIs(p, "positions") && equal(s, p->args[0])) return p->args[1];
    if (empty(p))
      return s->sort == Symbol::Sort::Axes ? call("empty_axes", {})
                                           : make(Symbol::Kind::List);
    if (callIs(p, "range") && equal(p->args[0], call("len", {s}))) return s;
    if (callIs(s, "range")) return p;
    if (callIs(s, "concat")) {
      auto offset = integer(0);
      for (const auto& part : s->args) {
        if (callIs(p, "range") && equal(offset, integer(0)) &&
            equal(p->args[0], call("len", {part})))
          return part;
        if (callIs(p, "offset_positions") && equal(p->args[1], offset) &&
            callIs(p->args[0], "range") &&
            equal(p->args[0]->args[0], call("len", {part})))
          return part;
        offset = call("+", {offset, call("len", {part})});
      }
    }
    // Preserve an extent lookup until the contextual theory can use
    // extents(A)[positions(A,B)] = extents(B).
    if (callIs(s, "gather") &&
        !(s->sort == Symbol::Sort::Extents && callIs(p, "positions")))
      return call("gather", {s->args[0], call("gather", {s->args[1], p})});
    if (p->kind == Symbol::Kind::List && p->args.empty()) return p;
  }
  if (name == "==" && equal(args[0], args[1])) return boolean(true);
  return make(Symbol::Kind::Call, std::move(name), 0, 0, std::move(args));
}
bool truth(const S& s) {
  return s->kind == Symbol::Kind::Boolean && s->name == "true";
}

}  // namespace tepl::core::index_semantics
