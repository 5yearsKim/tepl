#include "src/core/index_semantics/context.h"

namespace tepl::core::index_semantics {
S Context::canonical(const S& s) {
  ++counters.canonical;
  tick();
  const auto original = key(s);
  auto cached = queries_.canonical.find(original);
  if (cached != queries_.canonical.end()) {
    ++counters.cache_hits;
    return cached->second;
  }
  if (!queries_.active_canonical.insert(original).second) return s;
  ActiveKey active{queries_.active_canonical, original};
  if (queries_.active_canonical.size() > ProofLimits::expression_depth)
    throw ProofFailure("axis normalization depth limit exceeded",
                       FailureKind::BudgetExceeded);
  auto found = facts_.representatives.find(key(s));
  if (found != facts_.representatives.end()) return canonical(found->second);
  std::vector<S> args;
  for (const auto& a : s->args) args.push_back(canonical(a));
  S out = s;
  if (s->kind == Symbol::Kind::Call) {
    if (s->name == "gather" && callIs(args[1], "range") &&
        evidence(call("==", {args[1]->args[0], call("len", {args[0]})})))
      out = args[0];
    else if (s->name == "gather")
      out = selectKnown(args[0], args[1], QueryMode::Evidence);
    else if (s->name == "filter_axes")
      out = filter(args[0], support(args[1], QueryMode::Evidence),
                   QueryMode::Evidence);
    else if (s->name == "unique_axes" &&
             evidence(call("distinct_axes", {args[0]})))
      out = args[0];
    else if (s->name == "len" && callIs(args[0], "ordered_complement") &&
             callIs(args[0]->args[0], "range") &&
             evidence(call("is_valid_axis_list",
                           {args[0]->args[1], args[0]->args[0]->args[0]})))
      out = call("-",
                 {args[0]->args[0]->args[0], call("len", {args[0]->args[1]})});
    else if (s->name == "len" && callIs(args[0], "positions") &&
             evidence(call("distinct_axes", {args[0]->args[0]})) &&
             evidence(call("distinct_axes", {args[0]->args[1]})) &&
             evidence(call("subset", {args[0]->args[1], args[0]->args[0]})))
      out = call("len", {args[0]->args[1]});
    else if (s->name == "len" && callIs(args[0], "filter_axes") &&
             evidence(call("distinct_axes", {args[0]->args[0]})) &&
             evidence(call("distinct_axes", {args[0]->args[1]})) &&
             evidence(call("subset", {args[0]->args[1], args[0]->args[0]})))
      out = call("len", {args[0]->args[1]});
    else if (s->name == "inverse" && callIs(args[0], "inverse") &&
             evidence(
                 call("axis_partition",
                      {args[0]->args[0], call("len", {args[0]->args[0]})})))
      out = args[0]->args[0];
    else if (s->name == "inverse" && callIs(args[0], "gather") &&
             evidence(
                 call("axis_partition",
                      {args[0]->args[0], call("len", {args[0]->args[0]})})) &&
             evidence(
                 call("axis_partition",
                      {args[0]->args[1], call("len", {args[0]->args[0]})})))
      out = call("gather", {call("inverse", {args[0]->args[1]}),
                            call("inverse", {args[0]->args[0]})});
    else
      out = call(s->name, std::move(args));
    if (callIs(out, "ordered_complement") && callIs(out->args[0], "range") &&
        callIs(out->args[1], "concat") && out->args[1]->args.size() == 2) {
      auto excluded = out->args[1]->args;
      for (std::size_t i = 0; i < 2; ++i) {
        auto p = excluded[i], q = excluded[1 - i];
        auto retained = call("ordered_complement", {out->args[0], p});
        if (callIs(q, "gather") && equal(q->args[0], retained) &&
            evidence(call("is_valid_axis_list", {p, out->args[0]->args[0]})) &&
            evidence(call("is_valid_axis_list",
                          {q->args[1], call("len", {retained})})))
          out = call("gather",
                     {retained, call("ordered_complement",
                                     {call("range", {call("len", {retained})}),
                                      q->args[1]})});
        if (callIs(out, "gather")) break;
      }
    }
  } else if (!args.empty())
    out = make(s->kind, s->name, s->owner, s->index, std::move(args));
  found = facts_.representatives.find(key(out));
  auto result =
      found == facts_.representatives.end() ? out : canonical(found->second);
  if (key(result) != original) {
    // Cache the intermediate before another pass so shared DAGs and mutually
    // referring source facts cannot expand without a bound.
    queries_.canonical[original] = result;
    result = canonical(result);
  }
  queries_.canonical[original] = result;
  return result;
}
bool Context::elementary(const S& x) const {
  if (x->kind == Symbol::Kind::AxisGroup || callIs(x, "empty_axes"))
    return true;
  if (partitionPiece(x)) return true;
  if (callIs(x, "union_axes") || callIs(x, "concat")) {
    for (const auto& a : x->args)
      if (!elementary(a)) return false;
    return true;
  }
  return false;
}
namespace {
void tokens(const S& s, std::set<std::pair<std::size_t, std::size_t>>& out) {
  if (s->kind == Symbol::Kind::AxisGroup) {
    out.emplace(s->owner, s->index);
    return;
  }
  if (callIs(s, "intersection_axes") || callIs(s, "filter_axes")) {
    std::set<std::pair<std::size_t, std::size_t>> x, y;
    tokens(s->args[0], x);
    tokens(s->args[1], y);
    for (const auto& t : x)
      if (y.contains(t)) out.insert(t);
    return;
  }
  if (callIs(s, "difference_axes") || callIs(s, "gather")) {
    tokens(s->args[0], out);
    return;
  }
  for (const auto& a : s->args)
    if (a->sort == Symbol::Sort::Axes) tokens(a, out);
}
}  // namespace
bool Context::independent(const S& a, const S& b) {
  std::set<std::pair<std::size_t, std::size_t>> x, y;
  tokens(a, x);
  tokens(b, y);
  for (const auto& t : x)
    if (y.contains(t)) return false;
  return true;
}
S Context::support(const S& a, QueryMode mode) {
  tick();
  auto id = std::pair(mode, key(a));
  auto found = queries_.support.find(id);
  if (found != queries_.support.end()) return found->second;
  auto result = computeSupport(a, mode);
  queries_.support[id] = result;
  return result;
}
S Context::computeSupport(const S& a, QueryMode mode) {
  auto x = canonical(a);
  auto id = std::to_string(static_cast<int>(mode)) + ":" + key(x);
  if (!queries_.active_support.insert(id).second) return x;
  ActiveKey active{queries_.active_support, id};
  if (queries_.active_support.size() > ProofLimits::expression_depth)
    throw ProofFailure("axis support depth limit exceeded",
                       FailureKind::BudgetExceeded);
  if (auto found = bindings_.cover(x)) {
    auto covered = canonical(found);
    if (!equal(x, covered)) return support(covered, mode);
  }
  if (callIs(x, "unique_axes")) return support(x->args[0], mode);
  if (callIs(x, "difference_axes"))
    return difference(x->args[0], x->args[1], mode);
  if (callIs(x, "intersection_axes") || callIs(x, "filter_axes"))
    return intersect(x->args[0], x->args[1], mode);
  if (callIs(x, "gather") &&
      established(
          call("axis_partition", {x->args[1], call("len", {x->args[0]})})))
    return support(x->args[0], mode);
  if (callIs(x, "gather") &&
      query(call("is_valid_axis_list", {x->args[1], call("len", {x->args[0]})}),
            mode) &&
      query(call("==", {call("len", {x->args[1]}), call("len", {x->args[0]})}),
            mode))
    return support(x->args[0], mode);
  if (callIs(x, "concat") || callIs(x, "union_axes")) {
    std::vector<S> out;
    for (const auto& a : x->args) out.push_back(support(a, mode));
    return call("union_axes", std::move(out));
  }
  return x;
}
S Context::intersect(const S& a, const S& b, QueryMode mode) {
  tick();
  auto x = support(a, mode), y = support(b, mode);
  if (equal(x, y)) return x;
  if (independent(x, y)) return call("empty_axes", {});
  if (partitionDisjoint(x, y)) return call("empty_axes", {});
  if (!elementary(x) || !elementary(y)) {
    if (callIs(x, "intersection_axes") &&
        (equal(x->args[0], y) || equal(x->args[1], y)))
      return x;
    if (callIs(y, "intersection_axes") &&
        (equal(y->args[0], x) || equal(y->args[1], x)))
      return y;
    if (key(y) < key(x)) std::swap(x, y);
    return call("intersection_axes", {x, y});
  }
  if (callIs(x, "union_axes") || callIs(x, "concat")) {
    std::vector<S> out;
    for (const auto& p : x->args) out.push_back(intersect(p, y, mode));
    return call("union_axes", std::move(out));
  }
  if (callIs(y, "union_axes") || callIs(y, "concat"))
    return intersect(y, x, mode);
  return call("intersection_axes", {x, y});
}
S Context::difference(const S& a, const S& b, QueryMode mode) {
  tick();
  auto x = support(a, mode), y = support(b, mode);
  if (equal(x, y)) return call("empty_axes", {});
  if (independent(x, y)) return x;
  if (partitionDisjoint(x, y)) return x;
  if (!elementary(x) || !elementary(y)) return call("difference_axes", {x, y});
  if (callIs(x, "union_axes") || callIs(x, "concat")) {
    std::vector<S> out;
    for (const auto& p : x->args) out.push_back(difference(p, y, mode));
    return call("union_axes", std::move(out));
  }
  if (callIs(y, "union_axes") || callIs(y, "concat")) {
    auto out = x;
    for (const auto& p : y->args) out = difference(out, p, mode);
    return out;
  }
  return call("difference_axes", {x, y});
}
S Context::filter(const S& order, const S& wanted, QueryMode mode) {
  tick();
  auto w = support(wanted, mode), o = canonical(order);
  if (equal(o, w)) return o;
  if (callIs(o, "unique_axes"))
    return call("unique_axes", {filter(o->args[0], w, mode)});
  if (callIs(o, "concat")) {
    std::vector<S> out;
    for (const auto& p : o->args) out.push_back(filter(p, w, mode));
    return call("concat", std::move(out));
  }
  if (independent(o, w) || callIs(w, "empty_axes"))
    return call("empty_axes", {});
  if (equal(intersect(o, w, mode), support(o, mode))) return o;
  auto p = positions(o, w, mode);
  if (callIs(p, "ordered_complement") || callIs(p, "range"))
    return select(o, p, mode);
  return call("filter_axes", {o, w});
}
S Context::extents(const S& axes) {
  tick();
  if (axes->kind == Symbol::Kind::AxisGroup) return axes->args[0];
  if (callIs(axes, "empty_axes")) return make(Symbol::Kind::List);
  if (callIs(axes, "concat")) {
    std::vector<S> out;
    for (const auto& p : axes->args) out.push_back(extents(p));
    return call("concat", std::move(out));
  }
  if (callIs(axes, "gather"))
    return call("gather", {extents(axes->args[0]), axes->args[1]});
  return call("axis_extents", {axes});
}
}  // namespace tepl::core::index_semantics
