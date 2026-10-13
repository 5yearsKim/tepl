#include <algorithm>
#include <functional>
#include <tuple>

#include "src/core/index_semantics/context.h"

namespace tepl::core::index_semantics {
void Context::invalidate() {
  ++counters.invalidations;
  queries_.canonical.clear();
  queries_.entailment.clear();
  queries_.evidence.clear();
  queries_.support.clear();
  queries_.positions.clear();
}
void Context::learn(const S& fact) {
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
    auto ka = key(a), kb = key(b);
    if (ka == kb) return;
    std::function<std::size_t(const S&)> weight = [&](const S& x) {
      if (x->kind == Symbol::Kind::AxisGroup) return std::size_t(1);
      std::size_t n = 1;
      for (const auto& y : x->args) n += weight(y);
      return n;
    };
    auto category = [](const S& x) {
      if (x->kind == Symbol::Kind::Integer ||
          x->kind == Symbol::Kind::Boolean ||
          (x->kind == Symbol::Kind::List && x->args.empty()))
        return 0;
      return x->args.empty() || x->kind == Symbol::Kind::AxisGroup ? 1 : 2;
    };
    if (std::tuple(category(a), weight(a), ka) <
        std::tuple(category(b), weight(b), kb))
      std::swap(a, b);
    facts_.representatives[key(a)] = b;
  } else
    return;
  invalidate();
  // Only equality updates can rename existing premises. Refresh their index
  // here; adding a distinctness/range fact requires just one indexed insert.
  facts_.normalized.clear();
  for (const auto& raw : facts_.raw) {
    ++counters.fact_comparisons;
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
  if (callIs(f, "distinct_axes")) {
    auto a = f->args[0];
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
    return independent(a, b) || partitionDisjoint(a, b);
  }
  if (callIs(f, "same_set"))
    return equal(support(f->args[0], mode), support(f->args[1], mode));
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
  }
  if (callIs(f, "axis_partition")) {
    auto p = f->args[0];
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
    auto parts = [](const S& x) {
      std::vector<std::string> keys;
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

}  // namespace tepl::core::index_semantics
