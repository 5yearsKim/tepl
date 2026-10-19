#include <algorithm>
#include <functional>
#include <tuple>

#include "src/core/semantics/context.h"
#include "src/core/semantics/symbolic/index_expr.h"

namespace tepl::core::semantics {
void Context::represent(S a, S b) {
  if (equal(a, b)) return;
  std::function<std::pair<std::size_t, std::size_t>(const S&)> order =
      [&](const S& x) -> std::pair<std::size_t, std::size_t> {
    auto found = queries_.term_order.find(key(x));
    if (found != queries_.term_order.end()) return found->second;
    tick();
    std::size_t dependency = 0, weight = 1;
    if (x->kind == Symbol::Kind::Field &&
        source_.field_nodes.contains(x->owner))
      dependency = source_.field_nodes.at(x->owner) + 1;
    if (x->kind != Symbol::Kind::AxisGroup)
      for (const auto& y : x->args) {
        auto [d, w] = order(y);
        dependency = std::max(dependency, d);
        weight = std::min<std::size_t>(ProofLimits::general_inference_work,
                                       weight + w);
      }
    return queries_.term_order[key(x)] = {dependency, weight};
  };
  auto category = [](const S& x) {
    // Prefer executable shape/selector programs to opaque extent queries.
    // Otherwise a derived shape witness can rename its own constructor back
    // into the unresolved query and repeatedly expand it during composition.
    if (callIs(x, "axis_extents")) return 3;
    if (x->kind == Symbol::Kind::Integer || x->kind == Symbol::Kind::Boolean ||
        (x->kind == Symbol::Kind::List && x->args.empty()))
      return 0;
    return x->args.empty() || x->kind == Symbol::Kind::AxisGroup ? 1 : 2;
  };
  // Source nodes are topologically ordered. Prefer producer metadata over a
  // downstream field, so routing prerequisites cannot feed their own lengths
  // back into producer selectors. No operation or attribute names participate.
  auto ao = std::tuple(order(a).first, category(a), order(a).second);
  auto bo = std::tuple(order(b).first, category(b), order(b).second);
  if (ao < bo || (ao == bo && symbolLess(a, b))) std::swap(a, b);
  facts_.representatives[key(a)] = b;
}
S Context::canonicalChildren(const S& s) {
  if (s->kind != Symbol::Kind::Call && s->kind != Symbol::Kind::List)
    return canonical(s);
  std::vector<S> args;
  for (const auto& a : s->args) args.push_back(canonical(a));
  return s->kind == Symbol::Kind::Call
             ? call(s->name, std::move(args))
             : make(s->kind, s->name, s->owner, s->index, std::move(args));
}
void Context::invalidate() {
  ++counters.invalidations;
  queries_.canonical.clear();
  queries_.entailment.clear();
  queries_.evidence.clear();
  queries_.intervals.clear();
  queries_.uniform_elements.clear();
  queries_.support.clear();
  queries_.positions.clear();
  if (!active_sets_) {
    sets_.reset();
    set_expressions_.clear();
    set_atoms_.clear();
  }
}
void Context::learn(const S& fact) {
  if (callIs(fact, "&&")) {
    learn(fact->args[0]);
    learn(fact->args[1]);
    return;
  }
  if (truth(fact) || !facts_.keys.insert(key(fact)).second) return;
  auto f = canonical(fact);
  if (truth(f)) return;
  invalidate();
  facts_.normalized.insert(key(f));
  facts_.raw.push_back(fact);
  if (callIs(f, "is_valid_axis_list")) {
    facts_.valid_selectors[key(f->args[0])].push_back(f->args[1]);
    facts_.selectors[key(f->args[0])] = f->args[0];
    if (equal(canonical(call("len", {f->args[0]})), canonical(f->args[1]))) {
      facts_.permutations.insert(key(f->args[0]));
      invalidate();
    }
  }
  if (callIs(f, "axis_partition")) {
    facts_.permutations.insert(key(f->args[0]));
    learn(call("is_valid_axis_list", f->args));
    learn(call("==", {call("len", {f->args[0]}), f->args[1]}));
    return;
  }
  if (callIs(f, "same_set") &&
      (callIs(f->args[0], "empty_axes") || callIs(f->args[1], "empty_axes"))) {
    learn(call("==", f->args));
    return;
  }
  if (callIs(f, "==")) {
    auto a = f->args[0], b = f->args[1];
    // Sequence equality also establishes equal cardinality. The expression
    // factory lowers len(gather(shape,p)) to len(p), so this implication must
    // be registered before that lowering hides the paired-domain witness.
    if (a->sort == b->sort &&
        (a->sort == Symbol::Sort::Axes || a->sort == Symbol::Sort::Positions ||
         a->sort == Symbol::Sort::Extents))
      learn(call("==", {call("len", {a}), call("len", {b})}));
    represent(canonicalChildren(a), canonicalChildren(b));
  } else
    return;
  invalidate();
  // Only equality updates can rename existing premises. Refresh their index
  // here; adding a distinctness/range fact requires just one indexed insert.
  facts_.normalized.clear();
  for (const auto& raw : facts_.raw) {
    ++counters.fact_comparisons;
    if (callIs(raw, "==")) {
      // Preserve congruence when a new fact changes children of an already
      // recorded alias (for example, a selector containing a group length).
      represent(canonicalChildren(raw->args[0]),
                canonicalChildren(raw->args[1]));
    }
    auto normalized = canonical(raw);
    facts_.normalized.insert(key(normalized));
    if (callIs(normalized, "axis_partition"))
      facts_.permutations.insert(key(normalized->args[0]));
    if (callIs(normalized, "is_valid_axis_list")) {
      auto p = normalized->args[0], n = normalized->args[1];
      auto& bounds = facts_.valid_selectors[key(p)];
      if (std::none_of(bounds.begin(), bounds.end(),
                       [&](const S& x) { return equal(x, n); }))
        bounds.push_back(n);
      facts_.selectors[key(p)] = p;
    }
  }
  // Record permutation evidence at the mutation boundary. Map normalization
  // can subsequently use it without launching a proof of this property.
  for (const auto& [selector, domains] : facts_.valid_selectors)
    for (const auto& bound : domains) {
      if (equal(canonical(call("len", {facts_.selectors.at(selector)})),
                canonical(bound))) {
        facts_.permutations.insert(selector);
        break;
      }
    }
  invalidate();
}
bool Context::established(const S& f) {
  tick();
  // A previous evidence miss must not hide a subsequent successful proof.
  auto proved = queries_.entailment.find(key(f));
  if (proved != queries_.entailment.end() && proved->second) return true;
  auto found = queries_.evidence.find(key(f));
  if (found != queries_.evidence.end()) return found->second;
  auto result = deriveEvidence(f);
  queries_.evidence[key(f)] = result;
  return result;
}
bool Context::deriveEvidence(const S& f) {
  // Simplification consults established premises, never invokes entailment.
  if (truth(f) || facts_.keys.contains(key(f)) ||
      facts_.normalized.contains(key(f)))
    return true;
  if (!queries_.active_premises.insert(key(f)).second) return false;
  ActiveKey active{queries_.active_premises, key(f)};
  if (queries_.active_premises.size() > ProofLimits::evidence_depth)
    return false;
  return structuralEvidence(f, QueryMode::Evidence);
}
bool Context::structuralEvidence(const S& f, QueryMode mode) {
  if (callIs(f, "is_disjoint")) {
    auto a = f->args[0], b = f->args[1];
    if (facts_.normalized.contains(key(call("is_disjoint", {b, a}))))
      return true;
    auto subset = [&](auto&& self, const S& x, const S& y) -> bool {
      if (equal(x, y)) return true;
      if (callIs(x, "ordered_complement")) return self(self, x->args[0], y);
      if (callIs(x, "concat")) {
        for (const auto& p : x->args)
          if (!self(self, p, y)) return false;
        return true;
      }
      return x->kind == Symbol::Kind::List && x->args.empty();
    };
    for (const auto& [x, y] : {std::pair{a, b}, std::pair{b, a}})
      if (callIs(x, "ordered_complement") && subset(subset, y, x->args[1]))
        return true;
  }
  if (callIs(f, "<") || callIs(f, "<=") || callIs(f, ">") || callIs(f, ">="))
    return integerComparison(f, mode);
  if (callIs(f, "==") || callIs(f, "same_set") || callIs(f, "subset")) {
    auto empty = [](const S& x) {
      return callIs(x, "empty_axes") ||
             (x->kind == Symbol::Kind::List && x->args.empty());
    };
    auto sequence = [](const S& x) {
      return x->sort == Symbol::Sort::Axes ||
             x->sort == Symbol::Sort::Extents ||
             x->sort == Symbol::Sort::Positions;
    };
    auto a = f->args[0], b = f->args[1];
    // Scalars have a unique empty shape/interface. Prove this relationship
    // without replacing named coordinate views or losing their origin maps.
    if ((empty(a) || empty(b)) && sequence(a) && sequence(b) &&
        query(call("==", {call("len", {a}), integer(0)}), mode) &&
        query(call("==", {call("len", {b}), integer(0)}), mode))
      return true;
  }
  if (callIs(f, "&&"))
    return query(f->args[0], mode) && query(f->args[1], mode);
  if (callIs(f, "||"))
    return query(f->args[0], mode) || query(f->args[1], mode);
  // Scalar function signatures use the same dtype families as contracts.
  if (callIs(f, "is_numeric"))
    return query(call("is_float", f->args), mode) ||
           query(call("is_integer", f->args), mode);
  if (callIs(f, "is_integer"))
    return query(call("is_signed_integer", f->args), mode) ||
           query(call("is_unsigned_integer", f->args), mode);
  if (callIs(f, "extent_le") &&
      extentBoundArithmetic(*this, f->args[0], f->args[1], mode))
    return true;
  if (callIs(f, "is_valid_extents") && callIs(f->args[0], "offset_extents")) {
    auto d = f->args[0];
    if (d->args[1]->name == "1" && callIs(d->args[0], "extent_sub"))
      return query(
          call("extent_le", {d->args[0]->args[1], d->args[0]->args[0]}), mode);
  }

  if (callIs(f, "==") && f->args[0]->sort == Symbol::Sort::Extents &&
      f->args[1]->sort == Symbol::Sort::Extents) {
    auto left = shape_witnesses_.find(key(f->args[0]));
    auto right = shape_witnesses_.find(key(f->args[1]));
    if (left != shape_witnesses_.end() && right != shape_witnesses_.end()) {
      // Snapshot: proof queries must not iterate a table they can change.
      auto a = left->second, b = right->second;
      for (const auto& x : a)
        for (const auto& y : b)
          if (query(call("==", {x, y}), mode)) return true;
    }
  }
  if (callIs(f, "is_positive_extents")) {
    auto domain = canonical(f->args[0]);
    if (auto value = uniformElement(domain);
        value && truth(canonical(call(">", {value, integer(0)}))))
      return true;
    if (domain->kind == Symbol::Kind::List) {
      for (const auto& n : domain->args)
        if (!query(call(">", {n, integer(0)}), mode)) return false;
      return true;
    }
    if (callIs(domain, "extent_mul"))
      return query(call("==", {call("len", {domain->args[0]}),
                               call("len", {domain->args[1]})}),
                   mode) &&
             query(call("is_positive_extents", {domain->args[0]}), mode) &&
             query(call("is_positive_extents", {domain->args[1]}), mode);
    if (callIs(domain, "gather"))
      return query(call("is_positive_extents", {domain->args[0]}), mode) &&
             query(call("is_valid_axis_list",
                        {domain->args[1], call("len", {domain->args[0]})}),
                   mode);
    if (callIs(domain, "concat")) {
      for (const auto& part : domain->args)
        if (!query(call("is_positive_extents", {part}), mode)) return false;
      return true;
    }
  }
  if (callIs(f, "is_valid_extents")) {
    auto domain = canonical(f->args[0]);
    if (auto value = uniformElement(domain);
        value && truth(canonical(call(">=", {value, integer(0)}))))
      return true;
    if (query(call("is_positive_extents", {domain}), mode)) return true;
    if (callIs(domain, "extent_sub"))
      return query(call("extent_le", {domain->args[1], domain->args[0]}), mode);
    if (callIs(domain, "extent_floor_div"))
      return query(call("is_valid_extents", {domain->args[0]}), mode) &&
             query(call("is_positive_extents", {domain->args[1]}), mode);
    if (callIs(domain, "extent_mul") || callIs(domain, "extent_add"))
      return query(call("is_valid_extents", {domain->args[0]}), mode) &&
             query(call("is_valid_extents", {domain->args[1]}), mode);
    if (callIs(domain, "offset_extents") && domain->args[1]->name == "-1" &&
        query(call("is_positive_extents", {domain->args[0]}), mode))
      return true;
    if (callIs(domain, "offset_extents") && nonnegative(domain->args[1]))
      return query(call("is_valid_extents", {domain->args[0]}), mode);
    if (domain->kind == Symbol::Kind::Shape) return true;
    if (callIs(domain, "axis_extents")) {
      auto valid = [&](auto&& self, const S& axes) -> bool {
        if (axes->kind == Symbol::Kind::AxisGroup)
          return query(call("is_valid_extents", {axes->args[0]}), mode);
        if (callIs(axes, "empty_axes")) return true;
        if (callIs(axes, "gather"))
          return self(self, axes->args[0]) &&
                 query(call("is_valid_axis_list",
                            {axes->args[1], call("len", {axes->args[0]})}),
                       mode);
        if (callIs(axes, "filter_axes") || callIs(axes, "difference_axes") ||
            callIs(axes, "intersection_axes"))
          return self(self, axes->args[0]);
        if (callIs(axes, "place_axes"))
          return self(self, axes->args[0]) && self(self, axes->args[2]);
        if (callIs(axes, "concat") || callIs(axes, "union_axes")) {
          for (const auto& part : axes->args)
            if (!self(self, part)) return false;
          return true;
        }
        return false;
      };
      return valid(valid, domain->args[0]);
    }
    if (domain->kind == Symbol::Kind::List) {
      for (const auto& n : domain->args)
        if (!nonnegative(n) && !query(call(">=", {n, integer(0)}), mode))
          return false;
      return true;
    }
    if (callIs(domain, "concat")) {
      for (const auto& part : domain->args)
        if (!query(call("is_valid_extents", {part}), mode)) return false;
      return true;
    }
    if (callIs(domain, "gather"))
      return query(call("is_valid_extents", {domain->args[0]}), mode) &&
             query(call("is_valid_axis_list",
                        {domain->args[1], call("len", {domain->args[0]})}),
                   mode);
  }
  if (callIs(f, "distinct_axes")) {
    auto a = f->args[0];
    if (callIs(a, "place_axes"))
      return query(call("distinct_axes",
                        {call("concat", {a->args[0], a->args[2]})}),
                   mode) &&
             query(call("is_valid_axis_list", {a->args[1], call("len", {a})}),
                   mode) &&
             query(call("==",
                        {call("len", {a->args[1]}), call("len", {a->args[0]})}),
                   mode);
    if (a->kind == Symbol::Kind::AxisGroup || callIs(a, "empty_axes") ||
        callIs(a, "unique_axes"))
      return true;
    if (callIs(a, "concat")) {
      for (std::size_t i = 0; i < a->args.size(); ++i) {
        if (!query(call("distinct_axes", {a->args[i]}), mode)) return false;
        for (std::size_t j = 0; j < i; ++j)
          if (!query(call("disjoint", {a->args[i], a->args[j]}), mode))
            return false;
      }
      return true;
    }
    if (callIs(a, "gather"))
      return query(call("distinct_axes", {a->args[0]}), mode) &&
             query(call("is_valid_axis_list",
                        {a->args[1], call("len", {a->args[0]})}),
                   mode);
    if (callIs(a, "filter_axes") || callIs(a, "difference_axes") ||
        callIs(a, "intersection_axes"))
      return query(call("distinct_axes", {a->args[0]}), mode);
  }
  if (callIs(f, "disjoint")) {
    auto a = canonical(f->args[0]), b = canonical(f->args[1]);
    return independent(a, b) || partitionDisjoint(a, b) || setEntails(f);
  }
  if (callIs(f, "same_set"))
    return equal(support(f->args[0], mode), support(f->args[1], mode)) ||
           setEntails(f);
  if (callIs(f, "subset")) {
    auto a = support(f->args[0], mode), b = support(f->args[1], mode);
    if (equal(a, b) || callIs(a, "empty_axes")) return true;
    if (callIs(a, "intersection_axes") || callIs(a, "filter_axes"))
      return query(call("subset", {a->args[0], b}), mode) ||
             query(call("subset", {a->args[1], b}), mode);
    if (callIs(a, "difference_axes"))
      return query(call("subset", {a->args[0], b}), mode);
    if (callIs(a, "concat") || callIs(a, "union_axes")) {
      for (const auto& part : a->args)
        if (!query(call("subset", {part, b}), mode)) return false;
      return true;
    }
    if (callIs(b, "concat") || callIs(b, "union_axes")) {
      for (const auto& part : b->args)
        if (query(call("subset", {a, part}), mode)) return true;
    }
    return setEntails(f);
  }
  if (callIs(f, "axis_partition")) {
    auto p = f->args[0];
    if (callIs(p, "concat"))
      for (std::size_t i = 0; i < p->args.size(); ++i) {
        const auto& remainder = p->args[i];
        if (!callIs(remainder, "ordered_complement") ||
            !equal(canonical(remainder->args[0]),
                   canonical(call("range", {f->args[1]}))))
          continue;
        std::vector<S> selected;
        for (std::size_t j = 0; j < p->args.size(); ++j)
          if (i != j) selected.push_back(p->args[j]);
        auto positions = canonical(call("concat", std::move(selected)));
        if (equal(positions, canonical(remainder->args[1])) &&
            query(call("is_valid_axis_list", {positions, f->args[1]}), mode))
          return true;
      }
    if (callIs(p, "order_positions"))
      return query(call("==", {call("len", {p->args[0]}), f->args[1]}), mode);
    if (facts_.permutations.contains(key(p)) &&
        equal(canonical(call("len", {p})), canonical(f->args[1])))
      return true;
    if (callIs(p, "inverse"))
      return query(call("axis_partition", {p->args[0], f->args[1]}), mode);
    if (callIs(p, "gather"))
      return query(call("axis_partition", {p->args[0], f->args[1]}), mode) &&
             query(call("axis_partition",
                        {p->args[1], call("len", {p->args[0]})}),
                   mode);
    if (callIs(p, "concat") && p->args.size() == 2)
      for (std::size_t i = 0; i < 2; ++i) {
        auto c = p->args[i], r = p->args[1 - i];
        if (callIs(c, "ordered_complement") &&
            equal(canonical(c->args[0]),
                  canonical(call("range", {f->args[1]}))) &&
            equal(c->args[1], r) &&
            query(call("is_valid_axis_list", {r, f->args[1]}), mode))
          return true;
      }
  }
  if (callIs(f, "is_valid_axis_list")) {
    auto found = facts_.valid_selectors.find(key(f->args[0]));
    if (found != facts_.valid_selectors.end())
      for (const auto& n : found->second)
        if (equal(canonical(n), canonical(f->args[1]))) return true;
    auto p = f->args[0], n = f->args[1];
    if (callIs(p, "ordered_complement") &&
        query(call("is_valid_axis_list", {p->args[0], n}), mode))
      return true;
    if (callIs(p, "concat")) {
      bool valid = true;
      for (std::size_t i = 0; i < p->args.size(); ++i) {
        valid &= query(call("is_valid_axis_list", {p->args[i], n}), mode);
        for (std::size_t j = 0; j < i; ++j)
          valid &= query(call("is_disjoint", {p->args[i], p->args[j]}), mode);
      }
      if (valid) return true;
    }
    if (p->kind == Symbol::Kind::List && p->args.empty() && nonnegative(n))
      return true;
    if (auto block = positionBlock(p)) {
      auto end = canonical(call("+", {block->first, block->second}));
      auto bound = canonical(n);
      if (equal(end, bound)) return true;
      // A prefix/middle block in a sum of nonnegative axis-group sizes.
      std::vector<S> summands;
      std::function<void(const S&)> flatten = [&](const S& x) {
        if (callIs(x, "+")) {
          flatten(x->args[0]);
          flatten(x->args[1]);
        } else
          summands.push_back(x);
      };
      flatten(bound);
      auto prefix = integer(0);
      for (std::size_t i = 0; i < summands.size(); ++i) {
        prefix = canonical(call("+", {prefix, summands[i]}));
        if (!equal(prefix, end)) continue;
        bool valid = true;
        for (std::size_t j = i + 1; j < summands.size(); ++j)
          valid &= nonnegative(summands[j]);
        if (valid) return true;
      }
    }
    if (callIs(p, "order_positions"))
      return query(call("==", {call("len", {p->args[0]}), n}), mode);
    if (callIs(p, "matched_positions")) {
      // Matching a distinct list returns distinct positions in the second
      // list. A selection retains its parent's bound, so use that established
      // domain when the selected list has no separate premise yet.
      auto root = p->args[0];
      while (callIs(root, "gather")) root = root->args[0];
      auto bounds = facts_.valid_selectors.find(key(root));
      if (bounds == facts_.valid_selectors.end()) return false;
      const auto domains = bounds->second;
      for (const auto& domain : domains)
        if (query(call("is_valid_axis_list", {p->args[0], domain}), mode) &&
            query(call("==", {call("len", {p->args[1]}), n}), mode))
          return true;
      return false;
    }
    if (callIs(p, "membership_positions"))
      return query(call("==", {call("len", {p->args[0]}), n}), mode);
    if (callIs(p, "ordered_complement") &&
        equal(canonical(p->args[0]), canonical(call("range", {n}))))
      return true;
    if (callIs(p, "range")) return equal(canonical(p->args[0]), canonical(n));
    if (callIs(p, "inverse"))
      return query(call("axis_partition", {p->args[0], n}), mode);
    if (callIs(p, "gather"))
      return query(call("is_valid_axis_list", {p->args[0], n}), mode) &&
             query(call("is_valid_axis_list",
                        {p->args[1], call("len", {p->args[0]})}),
                   mode);
    if (callIs(p, "positions"))
      return query(call("distinct_axes", {p->args[0]}), mode) &&
             query(call("distinct_axes", {p->args[1]}), mode) &&
             query(call("subset", {p->args[1], p->args[0]}), mode) &&
             (mode == QueryMode::Evidence
                  ? equal(canonical(call("len", {p->args[0]})), canonical(n))
                  : entails(call("==", {call("len", {p->args[0]}), n})));
  }
  return false;
}
bool Context::evidence(const S& f) {
  if (established(f)) return true;
  auto normalized = canonical(f);
  return !equal(f, normalized) && established(normalized);
}
bool Context::query(const S& f, QueryMode mode) {
  return mode == QueryMode::Evidence ? evidence(f) : entails(f);
}
bool Context::entails(const S& f) {
  ++counters.queries;
  if (established(f)) return true;
  // Unresolved metadata gets a bounded entailment attempt. A failed attempt
  // remains a condition (or fails strict verification), rather than consuming
  // the entire rule budget on one recursive membership question.
  if (!queries_.active_queries.empty() &&
      (queries_.active_queries.size() >= ProofLimits::query_depth ||
       queries_.start_budget - budget_ > ProofLimits::query_work))
    return false;
  if (queries_.active_queries.empty()) queries_.start_budget = budget_;
  auto original = key(f);
  if (!queries_.active_queries.insert(original).second) return false;
  ActiveKey active{queries_.active_queries, original};
  auto c = canonical(f);
  if (established(c)) return true;
  auto k = key(c);
  auto found = queries_.entailment.find(k);
  if (found != queries_.entailment.end()) return found->second;
  bool result = prove(c);
  queries_.entailment[k] = result;
  return result;
}
bool Context::prove(const S& f) {
  auto c = canonical(f);
  if (truth(c) || facts_.normalized.contains(key(c)) ||
      structuralEvidence(c, QueryMode::Entailment))
    return true;
  if (callIs(c, "==") && callIs(c->args[0], "len") &&
      callIs(c->args[1], "len")) {
    auto a = c->args[0]->args[0], b = c->args[1]->args[0];
    if (a->sort == Symbol::Sort::Axes && b->sort == Symbol::Sort::Axes &&
        entails(call("distinct_axes", {a})) &&
        entails(call("distinct_axes", {b})) &&
        entails(call("same_set", {a, b})))
      return true;
    if (callIs(a, "ordered_complement") && callIs(b, "ordered_complement") &&
        callIs(a->args[0], "range") && equal(a->args[0], b->args[0])) {
      auto n = a->args[0]->args[0];
      if (entails(call("is_valid_axis_list", {a->args[1], n})) &&
          entails(call("is_valid_axis_list", {b->args[1], n})) &&
          entails(call("==",
                       {call("len", {a->args[1]}), call("len", {b->args[1]})})))
        return true;
    }
  }
  if (callIs(c, "axis_partition")) {
    if (callIs(c->args[0], "concat"))
      for (const auto& partition : bindings_.partitions())
        if (equal(canonical(call("len", {partition.interface})), c->args[1])) {
          auto selected = select(partition.interface, c->args[0]);
          if (!callIs(selected, "gather") &&
              entails(call("distinct_axes", {selected})) &&
              entails(call("same_set", {selected, partition.interface})))
            return true;
        }
    if (entails(call("is_valid_axis_list", c->args)) &&
        entails(call("==", {call("len", {c->args[0]}), c->args[1]})))
      return true;
    auto parts = [&](const S& x) {
      std::vector<Key> keys;
      if (callIs(x, "concat"))
        for (const auto& a : x->args) keys.push_back(key(a));
      else
        keys.push_back(key(x));
      std::sort(keys.begin(), keys.end());
      return keys;
    };
    for (const auto& fact : facts_.raw) {
      auto known = canonical(fact);
      if (callIs(known, "axis_partition") &&
          equal(known->args[1], c->args[1]) &&
          parts(known->args[0]) == parts(c->args[0]))
        return true;
    }
  }
  if (callIs(c, "disjoint"))
    return callIs(intersect(c->args[0], c->args[1]), "empty_axes");
  if (callIs(c, "subset"))
    return callIs(difference(c->args[0], c->args[1]), "empty_axes");
  if (callIs(c, "is_valid_axis_list") && callIs(c->args[0], "inverse")) {
    auto p = c->args[0]->args[0];
    return entails(call("is_valid_axis_list", {p, c->args[1]})) &&
           entails(call("==", {call("len", {p}), c->args[1]}));
  }
  if (callIs(c, "is_valid_axis_list")) {
    auto p = c->args[0], n = c->args[1];
    // The complete source partition supplies a coordinate witness for
    // composed position lists. A distinct selected sequence is in range.
    if (callIs(p, "concat"))
      for (const auto& partition : bindings_.partitions())
        if (equal(canonical(call("len", {partition.interface})), n)) {
          auto selected = select(partition.interface, p);
          if (!callIs(selected, "gather") &&
              entails(call("distinct_axes", {selected})))
            return true;
        }
  }
  return false;
}

}  // namespace tepl::core::semantics
