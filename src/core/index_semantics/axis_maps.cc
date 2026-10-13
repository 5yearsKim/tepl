#include "src/core/index_semantics/context.h"

namespace tepl::core::index_semantics {
void Context::bind(const S& interface, const S& group, const S& p) {
  if (bindings_.bind(interface, group, p)) invalidate();
}
void Context::cover(const S& interface, const S& groups) {
  if (bindings_.cover(interface, groups)) invalidate();
}
void Context::partition(const S& interface, const std::vector<S>& groups,
                        const std::vector<S>& selectors) {
  if (!bindings_.partition(interface, groups, selectors)) return;
  invalidate();
  auto existing = bindings_.cover(interface);
  if (!existing || equal(existing, interface))
    cover(interface, call("concat", groups));
  learn(call("distinct_axes", {interface}));
  learn(call(
      "==", {call("len", {interface}), call("len", {call("concat", groups)})}));
  for (std::size_t i = 0; i < groups.size(); ++i) {
    bind(interface, groups[i], selectors[i]);
    // A group can be traversed in its declared order or in the physical order
    // of this interface. The latter selects the complement of all other
    // groups, because this table is a complete, disjoint partition.
    std::vector<S> other;
    for (std::size_t j = 0; j < groups.size(); ++j)
      if (i != j) other.push_back(selectors[j]);
    bind(interface, call("filter_axes", {interface, groups[i]}),
         call("ordered_complement", {call("range", {call("len", {interface})}),
                                     call("concat", std::move(other))}));
    learn(call("==", {call("len", {groups[i]}), call("len", {selectors[i]})}));
    learn(call("is_valid_axis_list", {selectors[i], call("len", {interface})}));
    learn(call("distinct_axes", {groups[i]}));
    for (std::size_t j = 0; j < i; ++j)
      learn(call("disjoint", {groups[i], groups[j]}));
  }
}
bool Context::partitionPiece(const S& x) const {
  for (const auto& p : bindings_.partitions())
    for (const auto& g : p.groups)
      if (equal(x, g)) return true;
  return false;
}
bool Context::partitionDisjoint(const S& a, const S& b) {
  for (const auto& p : bindings_.partitions()) {
    for (std::size_t i = 0; i < p.groups.size(); ++i)
      if (equal(a, canonical(p.groups[i])))
        for (std::size_t j = 0; j < p.groups.size(); ++j)
          if (i != j && equal(b, canonical(p.groups[j]))) return true;
  }
  return false;
}
S Context::positions(const S& a, const S& b, QueryMode mode) {
  ++counters.positions;
  tick();
  auto id = std::tuple(mode, key(a), key(b));
  auto found = queries_.positions.find(id);
  if (found != queries_.positions.end()) return found->second;
  auto result = lookupPositions(a, b, mode);
  queries_.positions[id] = result;
  return result;
}
S Context::lookupPositions(const S& a, const S& b, QueryMode mode) {
  if (auto found = bindings_.positions(a, b)) return canonical(found);
  auto x = canonical(a), y = canonical(b);
  if (auto found = bindings_.positions(x, y)) return canonical(found);
  auto id =
      std::to_string(static_cast<int>(mode)) + ":" + key(x) + ":" + key(y);
  if (!queries_.active_positions.insert(id).second)
    return call("positions", {x, y});
  ActiveKey active{queries_.active_positions, id};
  if (queries_.active_positions.size() > ProofLimits::expression_depth)
    throw ProofFailure("position lookup depth limit exceeded",
                       FailureKind::BudgetExceeded);
  if (callIs(y, "empty_axes")) return make(Symbol::Kind::List);
  if (equal(x, y)) return canonical(call("range", {call("len", {x})}));
  if (callIs(y, "gather") && equal(x, y->args[0])) return y->args[1];
  for (auto it = bindings_.wiring().rbegin(); it != bindings_.wiring().rend();
       ++it)
    if (equal(x, canonical(it->interface)) && equal(y, canonical(it->group)))
      return canonical(it->positions);
  if (callIs(x, "gather") &&
      query(call("axis_partition", {x->args[1], call("len", {x->args[0]})}),
            mode))
    return call("gather", {call("inverse", {x->args[1]}),
                           positions(x->args[0], y, mode)});
  if (callIs(y, "concat")) {
    std::vector<S> ps;
    for (const auto& part : y->args) ps.push_back(positions(x, part, mode));
    return call("concat", std::move(ps));
  }
  if (callIs(x, "concat")) {
    S offset = integer(0);
    for (const auto& part : x->args) {
      if (equal(part, y))
        return call("offset_positions",
                    {call("range", {call("len", {y})}), offset});
      if (query(call("subset", {y, part}), mode))
        return call("offset_positions", {positions(part, y, mode), offset});
      offset = call("+", {offset, call("len", {part})});
    }
  }
  for (auto it = bindings_.partitions().rbegin();
       it != bindings_.partitions().rend(); ++it)
    if (equal(x, canonical(it->interface)))
      for (std::size_t i = 0; i < it->groups.size(); ++i)
        if (!equal(x, canonical(it->groups[i])) &&
            query(call("subset", {y, it->groups[i]}), mode))
          return call("gather",
                      {it->selectors[i], positions(it->groups[i], y, mode)});
  return call("positions", {x, y});
}
S Context::select(const S& a, const S& p, QueryMode mode) {
  tick();
  auto x = canonical(a), q = canonical(p);
  return canonical(selectKnown(x, q, mode));
}
void Context::selectionFacts(const S& a, const S& p, const S& result) {
  auto x = canonical(a), q = canonical(p);
  if (x->sort == Symbol::Sort::Axes &&
      entails(call("is_valid_axis_list", {q, call("len", {x})}))) {
    // Extents and cardinality follow selection, including partial selections.
    // These are consequences of the sequence law, not RHS legality premises.
    learn(call("==",
               {extents(result), call("gather", {canonical(extents(x)), q})}));
    learn(call("==", {call("len", {result}), call("len", {q})}));
    if (entails(call("distinct_axes", {x}))) {
      learn(call("distinct_axes", {result}));
      bind(x, result, q);
    }
  }
}
S Context::selectKnown(const S& x, const S& q, QueryMode mode) {
  tick();
  if (auto found = bindings_.group(x, q)) return canonical(found);
  auto permutation = [&](const S& p) {
    return query(call("axis_partition", {p, call("len", {p})}), mode);
  };
  if (callIs(x, "inverse") && callIs(x->args[0], "concat") &&
      permutation(x->args[0])) {
    auto offset = integer(0);
    for (const auto& part : x->args[0]->args) {
      if (equal(q, part))
        return call("offset_positions",
                    {call("range", {call("len", {part})}), offset});
      if (callIs(q, "gather") && equal(q->args[0], part) &&
          query(call("is_valid_axis_list", {q->args[1], call("len", {part})}),
                mode))
        return call("offset_positions", {q->args[1], offset});
      offset = call("+", {offset, call("len", {part})});
    }
  }
  if (callIs(x, "inverse") && callIs(q, "gather") &&
      equal(x->args[0], q->args[0]) && permutation(x->args[0]))
    return q->args[1];
  if (callIs(q, "gather") && callIs(q->args[0], "inverse") &&
      equal(x, q->args[0]->args[0]) && permutation(x))
    return q->args[1];
  if (callIs(q, "inverse") && equal(x, q->args[0]) && permutation(x))
    return call("range", {call("len", {x})});
  if (callIs(q, "concat")) {
    std::vector<S> groups;
    for (const auto& part : q->args) groups.push_back(select(x, part, mode));
    return call("concat", std::move(groups));
  }
  if (callIs(x, "concat") && callIs(q, "offset_positions")) {
    auto offset = integer(0);
    for (const auto& part : x->args) {
      if (equal(q->args[1], offset) &&
          query(call("is_valid_axis_list", {q->args[0], call("len", {part})}),
                mode))
        return select(part, q->args[0], mode);
      offset = call("+", {offset, call("len", {part})});
    }
  }
  if (x->sort == Symbol::Sort::Extents && callIs(q, "positions") &&
      query(call("distinct_axes", {q->args[0]}), mode) &&
      query(call("distinct_axes", {q->args[1]}), mode) &&
      query(call("subset", {q->args[1], q->args[0]}), mode) &&
      equal(x, canonical(extents(q->args[0]))))
    return canonical(extents(q->args[1]));
  if (x->sort != Symbol::Sort::Axes) return call("gather", {x, q});
  for (auto it = bindings_.wiring().rbegin(); it != bindings_.wiring().rend();
       ++it)
    if (equal(x, it->interface) && equal(q, canonical(it->positions)))
      return canonical(it->group);
  if (callIs(q, "positions") && equal(x, q->args[0])) return q->args[1];
  if (callIs(q, "ordered_complement") &&
      equal(q->args[0], canonical(call("range", {call("len", {x})}))) &&
      query(call("is_valid_axis_list", {q->args[1], call("len", {x})}), mode)) {
    auto removed = select(x, q->args[1], mode);
    return call("filter_axes", {x, difference(x, removed, mode)});
  }
  for (auto it = bindings_.partitions().rbegin();
       it != bindings_.partitions().rend(); ++it)
    if (equal(x, it->interface))
      for (std::size_t i = 0; i < it->selectors.size(); ++i)
        if (callIs(q, "gather") &&
            equal(q->args[0], canonical(it->selectors[i])))
          return select(it->groups[i], q->args[1], mode);
  return call("gather", {x, q});
}
S Context::replace(const S& axes, const S& from, const S& to) {
  tick();
  if (equal(axes, from)) return to;
  if (independent(axes, from)) return axes;
  if (callIs(axes, "concat")) {
    std::vector<S> out;
    for (const auto& a : axes->args) out.push_back(replace(a, from, to));
    return call("concat", std::move(out));
  }
  if (callIs(axes, "gather"))
    return select(replace(axes->args[0], from, to), axes->args[1]);
  auto p = positions(from, axes);
  if (!callIs(p, "positions")) return select(to, p);
  return call("replace_axes", {axes, from, to});
}
}  // namespace tepl::core::index_semantics
