#include "src/core/semantics/context.h"

namespace tepl::core::semantics {
AxisSets::Id Context::setExpression(const S& value) {
  tick();
  auto x = value;
  auto k = key(x);
  if (auto it = set_expressions_.find(k); it != set_expressions_.end())
    return it->second;
  AxisSets::Id result;
  if (auto overlap = bindings_.overlapFor(x)) {
    // A full overlap may be its own parent. Such a view is an atom
    // constrained by inclusion, rather than a recursively expanded formula.
    auto placeholder = sets_->atom(x);
    set_expressions_[k] = placeholder;
    result = sets_->conjunction(setExpression(overlap->left),
                                setExpression(overlap->right));
    sets_->assume(sets_->equivalent(placeholder, result));
    // Materialize only a requested overlap. Its membership selector is
    // intrinsically valid for the established parent coordinate sequence.
    auto previous = bindings_.splits().size();
    auto split = bindings_.split(overlap->left, overlap->slots);
    if (bindings_.splits().size() != previous) invalidate();
    set_expressions_[key(split.remaining)] =
        sets_->conjunction(setExpression(split.parent), sets_->negate(result));
  } else if (callIs(x, "empty_axes"))
    result = 0;
  else if (callIs(x, "concat") || callIs(x, "union_axes")) {
    result = 0;
    for (const auto& a : x->args)
      result = sets_->disjunction(result, setExpression(a));
  } else if (callIs(x, "unique_axes"))
    result = setExpression(x->args[0]);
  else if (callIs(x, "intersection_axes") || callIs(x, "filter_axes"))
    result = sets_->conjunction(setExpression(x->args[0]),
                                setExpression(x->args[1]));
  else if (callIs(x, "difference_axes"))
    result = sets_->conjunction(setExpression(x->args[0]),
                                sets_->negate(setExpression(x->args[1])));
  else {
    result = sets_->atom(x);
    // Register before descending into a containing interface.
    set_expressions_[k] = result;
    if (x->kind == Symbol::Kind::AxisGroup) {
      auto container = [&](const S& root) {
        if (auto covered = bindings_.cover(root);
            covered && !equal(root, covered))
          return true;
        for (const auto& raw : facts_.raw)
          if (callIs(raw, "==") && raw->args[0]->sort == Symbol::Sort::Axes &&
              (equal(raw->args[0], root) || equal(raw->args[1], root)))
            return true;
        return false;
      };
      if (!container(x))
        for (const auto& atom : set_atoms_)
          if (!container(atom))
            sets_->assume(
                sets_->negate(sets_->conjunction(result, sets_->atom(atom))));
      set_atoms_.push_back(x);
    } else if (callIs(x, "gather")) {
      if (callIs(x->args[0], "gather")) {
        auto inner = x->args[0];
        auto flat = call(
            "gather",
            {inner->args[0], call("gather", {inner->args[1], x->args[1]})});
        sets_->assume(sets_->equivalent(result, setExpression(flat)));
      }
      if (auto known = bindings_.group(x->args[0], x->args[1]);
          known && !equal(known, x))
        sets_->assume(sets_->equivalent(result, setExpression(known)));

      auto base = setExpression(x->args[0]);
      sets_->assume(sets_->disjunction(sets_->negate(result), base));
      auto p = x->args[1];
      // Complement selections retain exactly the unselected coordinates.
      // Recognize them also through an established permutation, without
      // invoking sequence entailment from a membership query.
      auto permutation = [&](const S& selector) {
        return facts_.permutations.contains(key(selector)) ||
               (callIs(selector, "inverse") &&
                facts_.permutations.contains(key(selector->args[0])));
      };
      S complement = p, traversal = call("range", {call("len", {x->args[0]})});
      if (callIs(p, "gather") && permutation(p->args[0])) {
        traversal = p->args[0];
        complement = p->args[1];
      }
      if (callIs(complement, "ordered_complement") &&
          equal(complement->args[0],
                call("range", {call("len", {traversal})}))) {
        auto removed = call(
            "gather",
            {x->args[0], call("gather", {traversal, complement->args[1]})});
        sets_->assume(sets_->equivalent(
            result,
            sets_->conjunction(base, sets_->negate(setExpression(removed)))));
      }
      if (callIs(p, "gather") &&
          (facts_.permutations.contains(key(p->args[1])) ||
           (callIs(p->args[1], "order_positions") &&
            equal(p->args[0], p->args[1]->args[0]))))
        sets_->assume(sets_->equivalent(
            result, setExpression(call("gather", {x->args[0], p->args[0]}))));
      if (facts_.permutations.contains(key(x->args[1])))
        sets_->assume(sets_->equivalent(result, base));
    }
  }
  return set_expressions_[k] = result;
}

S Context::simplifySet(const S& x) {
  if (active_sets_ || !sets_ready_) return x;
  for (const auto& [id, group] : source_.groups)
    if (setEntails(call("same_set", {x, group}))) return group;
  for (const auto& overlap : bindings_.overlaps())
    if (setEntails(call("same_set", {x, overlap.view}))) return overlap.view;
  return x;
}

bool Context::setEntails(const S& f) {
  if (!sets_ready_ || active_sets_ || f->args.size() != 2 ||
      f->args[0]->sort != Symbol::Sort::Axes ||
      f->args[1]->sort != Symbol::Sort::Axes)
    return false;
  struct Active {
    bool& flag;
    explicit Active(bool& f) : flag(f) { flag = true; }
    ~Active() { flag = false; }
  } active(active_sets_);
  if (!sets_) {
    sets_ = std::make_unique<AxisSets>([this] { tick(); });
    // Complete, established partitions provide both coverage and disjointness.
    // Merely proposed RHS partitions are never present in this table.
    for (const auto& p : bindings_.partitions()) {
      auto base = setExpression(p.interface);
      AxisSets::Id covered = 0;
      for (const auto& g : p.groups) {
        auto part = setExpression(g);
        sets_->assume(sets_->negate(sets_->conjunction(covered, part)));
        covered = sets_->disjunction(covered, part);
      }
      sets_->assume(sets_->equivalent(base, covered));
    }
    const auto splits = bindings_.splits();
    for (const auto& split : splits) {
      auto parent = setExpression(split.parent);
      auto selected = setExpression(split.selected);
      auto remaining = setExpression(split.remaining);
      sets_->assume(
          sets_->equivalent(parent, sets_->disjunction(selected, remaining)));
      sets_->assume(sets_->negate(sets_->conjunction(selected, remaining)));
    }
    for (const auto& raw : facts_.raw) {
      if (raw->args.size() != 2 || raw->args[0]->sort != Symbol::Sort::Axes ||
          raw->args[1]->sort != Symbol::Sort::Axes)
        continue;
      auto a = setExpression(raw->args[0]), b = setExpression(raw->args[1]);
      if (callIs(raw, "==") || callIs(raw, "same_set"))
        sets_->assume(sets_->equivalent(a, b));
      else if (callIs(raw, "disjoint"))
        sets_->assume(sets_->negate(sets_->conjunction(a, b)));
      else if (callIs(raw, "subset"))
        sets_->assume(sets_->disjunction(sets_->negate(a), b));
    }
  }
  auto a = setExpression(f->args[0]), b = setExpression(f->args[1]);
  if (callIs(f, "same_set")) return sets_->proves(sets_->equivalent(a, b));
  if (callIs(f, "disjoint"))
    return sets_->proves(sets_->negate(sets_->conjunction(a, b)));
  if (callIs(f, "subset"))
    return sets_->proves(sets_->disjunction(sets_->negate(a), b));
  return false;
}
}  // namespace tepl::core::semantics
