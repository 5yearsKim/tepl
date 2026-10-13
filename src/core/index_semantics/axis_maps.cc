#include <functional>

#include "src/core/index_semantics/context.h"

namespace tepl::core::index_semantics {
AxisPartition Context::split(const S& parent, const S& selector) {
  require(call("is_valid_axis_list", {selector, call("len", {parent})}));
  auto result = bindings_.split(parent, selector);
  invalidate();
  return result;
}
TensorAxisInterface Context::tensorInterface(const S& axes) const {
  return bindings_.tensorInterface(axes);
}
S Context::mappedPositions(const S& interface, const S& group) const {
  return bindings_.lookup(interface, group);
}
S Context::membershipPositions(const S& p, const S& q) {
  return matchPositions(p, q).left_positions;
}
void Context::bind(const S& interface, const S& group, const S& p) {
  if (bindings_.bind(interface, group, p)) invalidate();
}
void Context::cover(const S& interface, const S& groups) {
  // A view partition can mention its own parent through selections. Keep
  // that relation in the partition table, not as a recursive support alias.
  std::function<bool(const S&)> references = [&](const S& x) {
    if (equal(x, interface)) return true;
    for (const auto& child : x->args)
      if (references(child)) return true;
    return false;
  };
  if (references(groups)) return;
  if (bindings_.cover(interface, groups)) invalidate();
}
void Context::partition(const S& interface, const std::vector<S>& groups,
                        const std::vector<S>& selectors, bool refine) {
  auto prior = bindings_.partitionsFor(interface);
  if (!bindings_.partition(interface, groups, selectors)) return;
  invalidate();
  auto existing = bindings_.cover(interface);
  if (!existing || equal(existing, interface))
    cover(interface, call("concat", groups));
  if (!refine) {
    // Target legality has already been proved. Index its established maps
    // without feeding large constructor expressions into premise aliases.
    for (std::size_t i = 0; i < groups.size(); ++i) {
      bind(interface, groups[i], selectors[i]);
      learn(
          call("is_valid_axis_list", {selectors[i], call("len", {interface})}));
    }
    return;
  }
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
    auto physical =
        call("ordered_complement", {call("range", {call("len", {interface})}),
                                    call("concat", std::move(other))});
    bind(interface, call("filter_axes", {interface, groups[i]}), physical);
    auto order = call("order_positions", {selectors[i]});
    // A complete legal partition relates physical traversal to the declared
    // paired order. The ordering permutation itself is always a bijection.
    learn(call("==", {physical, call("gather", {selectors[i], order})}));
    learn(call("==", {call("gather", {physical, call("inverse", {order})}),
                      selectors[i]}));
    bind(interface, call("gather", {groups[i], order}), physical);
    learn(call("==", {call("len", {groups[i]}), call("len", {selectors[i]})}));
    learn(call("is_valid_axis_list", {selectors[i], call("len", {interface})}));
    learn(call("distinct_axes", {groups[i]}));
    for (std::size_t j = 0; j < i; ++j)
      learn(call("disjoint", {groups[i], groups[j]}));
  }
  // Refine only two established partitions of this interface. Keep both
  // coordinate witnesses, including the order of paired contraction slots.
  for (const auto& old : prior) {
    bool ordered_blocks = true;
    S offset = integer(0);
    for (const auto& selector : old.selectors) {
      auto block = positionBlock(selector);
      ordered_blocks &=
          block && equal(canonical(block->first), canonical(offset));
      if (block) offset = call("+", {offset, block->second});
    }
    for (std::size_t i = 0; i < groups.size(); ++i) {
      auto left = canonical(groups[i]);
      std::vector<S> ordered_parts;
      for (std::size_t j = 0; j < old.groups.size(); ++j) {
        auto right = canonical(old.groups[j]);
        if (equal(left, right)) {
          ordered_parts.push_back(left);
          continue;
        }
        if (independent(left, right) || partitionDisjoint(left, right))
          continue;
        auto slots = membershipPositions(selectors[i], old.selectors[j]);
        auto view = call("gather", {left, slots});
        ordered_parts.push_back(view);
        auto paired =
            matchPositions(selectors[i], old.selectors[j]).right_positions;
        if (bindings_.overlap(left, right, view, slots, paired)) invalidate();
        note("ordered-axis-split");
        note("paired-axis-selection");
        bind(left, view, slots);
        bind(interface, view, call("gather", {selectors[i], slots}));
        bind(right, view, paired);
      }
      // A stable selector traverses consecutive producer blocks in order.
      // This equality is geometry, not an algebraic equivalence assumption.
      bool stable = callIs(selectors[i], "ordered_complement") ||
                    callIs(selectors[i], "range");
      if (stable && ordered_blocks &&
          bindings_.orderedView(left, ordered_parts)) {
        invalidate();
        note("ordered-partition-traversal");
      }
    }
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
  if (auto found = bindings_.lookup(a, b)) return found;
  auto x = canonical(a), y = canonical(b);
  if (auto found = bindings_.lookup(x, y)) return found;
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
  if (callIs(y, "gather")) {
    const auto& base = y->args[0];
    auto map = bindings_.positions(x, base);
    if (map) return call("gather", {map, y->args[1]});
  }
  const auto wiring = bindings_.wiring();
  for (auto it = wiring.rbegin(); it != wiring.rend(); ++it)
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
  if (auto found = bindings_.group(x, q)) return found;
  return canonical(selectKnown(x, q, mode));
}
void Context::selectionFacts(const S& a, const S& p, const S& result) {
  auto x = canonical(a), q = canonical(p);
  if (x->sort == Symbol::Sort::Axes &&
      entails(call("is_valid_axis_list", {q, call("len", {x})}))) {
    auto domain = canonical(call("gather", {canonical(extents(x)), q}));
    // Extents and cardinality follow selection, including partial selections.
    // These are consequences of the sequence law, not RHS legality premises.
    learn(call("==", {extents(result), domain}));
    selected_extents_.emplace_back(result, domain);
    learn(call("==", {call("len", {result}), call("len", {q})}));
    if (entails(call("distinct_axes", {x}))) {
      learn(call("distinct_axes", {result}));
      bind(x, result, q);
      if (callIs(q, "ordered_complement") &&
          equal(canonical(q->args[0]),
                canonical(call("range", {call("len", {x})}))) &&
          entails(call("is_valid_axis_list", {q->args[1], call("len", {x})}))) {
        auto removed = select(x, q->args[1]);
        bind(x, removed, q->args[1]);
        auto rest = call("ordered_complement",
                         {call("range", {call("len", {x})}), q->args[1]});
        bind(x, result, rest);
        bindings_.split(x, q->args[1]);
        invalidate();
      }
      // Restrict established source maps to this selected interface. The
      // inclusion premise is proved before publishing the partial inverse.
      // Only source groups and requested overlaps are considered, never all
      // combinations of symbolic axes.
      std::vector<S> candidates;
      for (const auto& [id, group] : source_.groups)
        candidates.push_back(group);
      for (const auto& overlap : bindings_.overlaps())
        candidates.push_back(overlap.view);
      struct SurvivingMap {
        S group, origin, positions;
      };
      std::vector<SurvivingMap> surviving;
      std::set<std::string> seen;
      for (const auto& group : candidates) {
        if (!seen.insert(key(group)).second || equal(group, result)) continue;
        auto mapping = bindings_.lookup(x, group);
        if (!mapping || !setEntails(call("subset", {group, result}))) continue;
        surviving.push_back(
            {group, mapping, AxisBindings::restrict(mapping, q)});
      }
      for (const auto& [group, origin, mapping] : surviving) {
        note("axis-map-restriction");
        bind(result, group, mapping);
        learn(call("==", {AxisBindings::compose(q, mapping), origin}));
        // Inclusion plus the established injective parent maps makes this
        // a total map of the surviving group, including its cardinality.
        learn(call("==", {call("len", {mapping}), call("len", {group})}));
        learn(call("is_valid_axis_list", {mapping, call("len", {result})}));
      }
    }
  }
}
S Context::selectKnown(const S& x, const S& q, QueryMode mode) {
  tick();
  if (auto found = bindings_.group(x, q)) return canonical(found);
  if (callIs(q, "positions") && equal(x, q->args[0]) &&
      query(call("subset", {q->args[1], x}), mode) &&
      query(call("distinct_axes", {x}), mode))
    return q->args[1];
  if (callIs(q, "range") &&
      query(call("==", {q->args[0], call("len", {x})}), mode))
    return x;
  auto inserted = [&](auto&& self, const S& a) -> bool {
    if (callIs(a, "place_axes")) return true;
    if (callIs(a, "gather")) return self(self, a->args[0]);
    return false;
  };
  if (inserted(inserted, x) && x->sort == Symbol::Sort::Axes &&
      callIs(q, "ordered_complement") &&
      equal(q->args[0], canonical(call("range", {call("len", {x})}))) &&
      query(call("is_valid_axis_list", {q->args[1], call("len", {x})}), mode)) {
    auto removed = select(x, q->args[1], mode);
    return call("filter_axes", {x, difference(x, removed, mode)});
  }
  auto permutation = [&](const S& p) {
    return query(call("axis_partition", {p, call("len", {p})}), mode);
  };
  if (x->sort == Symbol::Sort::Axes && callIs(x, "gather") &&
      query(call("axis_partition", {x->args[1], call("len", {x->args[0]})}),
            mode) &&
      query(call("is_valid_axis_list", {q, call("len", {x})}), mode))
    return call("gather", {x->args[0], call("gather", {x->args[1], q})});
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
  if (callIs(x, "concat") && q->kind == Symbol::Kind::List) {
    auto offset = integer(0);
    for (const auto& part : x->args) {
      auto size = canonical(call("len", {part}));
      if (equal(q, canonical(call("offset_positions",
                                  {call("range", {size}), offset}))))
        return part;
      offset = canonical(call("+", {offset, size}));
    }
  }
  if (callIs(x, "concat") && callIs(q, "offset_positions")) {
    auto offset = integer(0);
    for (const auto& part : x->args) {
      if (equal(canonical(q->args[1]), canonical(offset)) &&
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
  const auto wiring = bindings_.wiring();
  for (auto it = wiring.rbegin(); it != wiring.rend(); ++it)
    if (equal(x, it->interface) && equal(q, canonical(it->positions)))
      return canonical(it->group);
  if (callIs(q, "positions") && equal(x, q->args[0])) return q->args[1];
  if (x->sort == Symbol::Sort::Axes && callIs(q, "ordered_complement") &&
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
