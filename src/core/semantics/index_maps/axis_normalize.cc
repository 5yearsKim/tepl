#include <algorithm>
#include <charconv>
#include <functional>
#include <limits>

#include "src/core/semantics/context.h"
#include "src/core/semantics/symbolic/metadata_sequences.h"

namespace tepl::core::semantics {
bool Context::orderedPositions(const S& p) const {
  if (p->kind == Symbol::Kind::List) {
    long previous = -1;
    for (const auto& v : p->args) {
      if (v->kind != Symbol::Kind::Integer) return false;
      long next;
      auto r = std::from_chars(v->name.data(), v->name.data() + v->name.size(),
                               next);
      if (r.ec != std::errc{} || next <= previous) return false;
      previous = next;
    }
    return true;
  }
  if (callIs(p, "range") || callIs(p, "membership_positions")) return true;
  if (callIs(p, "ordered_complement") || callIs(p, "offset_positions"))
    return orderedPositions(p->args[0]);
  if (callIs(p, "gather") || callIs(p, "matched_positions"))
    return orderedPositions(p->args[0]) && orderedPositions(p->args[1]);
  return false;
}
bool Context::nonnegative(const S& s) {
  auto n = canonical(s);
  if (callIs(n, "len")) return true;
  if (n->kind == Symbol::Kind::Integer) {
    std::int64_t value;
    auto parsed =
        std::from_chars(n->name.data(), n->name.data() + n->name.size(), value);
    return parsed.ec == std::errc{} && value >= 0;
  }
  if (callIs(n, "index")) {
    auto interval = queries_.intervals.find(key(n));
    return interval != queries_.intervals.end() && interval->second.first >= 0;
  }
  if (callIs(n, "+")) {
    for (const auto& a : n->args)
      if (!nonnegative(a)) return false;
    return true;
  }
  if (callIs(n, "-") && n->args.size() == 2 && callIs(n->args[1], "len") &&
      evidence(call("is_valid_axis_list", {n->args[1]->args[0], n->args[0]})))
    return true;
  // Factory simplifications may hide a sequence-length witness. All indexed
  // source selectors are defined under legal LHS metadata.
  for (const auto& [id, selector] : facts_.selectors)
    if (equal(n, canonical(call("len", {selector})))) return true;
  return false;
}
std::optional<std::pair<S, S>> Context::positionBlock(const S& p,
                                                      bool ordered) {
  if (callIs(p, "range") && nonnegative(p->args[0]))
    return std::pair(integer(0), p->args[0]);
  if (callIs(p, "offset_positions") && nonnegative(p->args[1]))
    if (auto block = positionBlock(p->args[0], ordered))
      return std::pair(canonical(call("+", {p->args[1], block->first})),
                       block->second);
  if (callIs(p, "concat") && !p->args.empty()) {
    auto first = positionBlock(p->args[0], ordered);
    if (!first) return {};
    auto [start, size] = *first;
    for (std::size_t i = 1; i < p->args.size(); ++i) {
      auto next = positionBlock(p->args[i], ordered);
      if (!next || !equal(canonical(call("+", {start, size})), next->first))
        return {};
      size = canonical(call("+", {size, next->second}));
    }
    return std::pair(start, size);
  }
  auto size = canonical(call("len", {p}));
  if ((!ordered || orderedPositions(p)) && nonnegative(size) &&
      evidence(call("axis_partition", {p, size})))
    return std::pair(integer(0), size);
  return {};
}
S Context::complementBlocks(const S& universe, const S& excluded) {
  if (!callIs(universe, "range"))
    return call("ordered_complement", {universe, excluded});
  // Work with sets of whole intervals; keep sequence ordering outside a
  // complement untouched. Symbolic interval endpoints must have witnesses.
  std::vector<std::pair<S, S>> blocks;
  auto parts =
      callIs(excluded, "concat") ? excluded->args : std::vector<S>{excluded};
  for (const auto& part : parts) {
    auto block = positionBlock(part);
    if (!block) return call("ordered_complement", {universe, excluded});
    blocks.push_back(*block);
  }
  auto offset = integer(0);
  // Merge adjacent intervals in whichever order they were declared.
  while (!blocks.empty()) {
    auto found = std::find_if(blocks.begin(), blocks.end(), [&](const auto& b) {
      return equal(canonical(b.first), canonical(offset));
    });
    if (found == blocks.end()) break;
    offset = canonical(call("+", {offset, found->second}));
    blocks.erase(found);
  }
  // Recognize the remaining suffix/middle interval from a symbolic sum.
  auto total = canonical(universe->args[0]);
  std::vector<S> sizes;
  std::function<void(const S&)> flatten = [&](const S& x) {
    if (callIs(x, "+")) {
      flatten(x->args[0]);
      flatten(x->args[1]);
    } else
      sizes.push_back(x);
  };
  flatten(total);
  for (std::size_t first = 0; first <= sizes.size(); ++first)
    for (std::size_t last = first; last <= sizes.size(); ++last) {
      auto sum = [&](std::size_t a, std::size_t b) {
        auto out = integer(0);
        for (auto i = a; i < b; ++i) out = call("+", {out, sizes[i]});
        return canonical(out);
      };
      auto start = sum(0, first), size = sum(first, last), end = sum(0, last);
      if (!equal(canonical(offset), start) || !nonnegative(size)) continue;
      if (blocks.empty() && last == sizes.size())
        return call("offset_positions", {call("range", {size}), start});
      if (blocks.size() == 1 && equal(canonical(blocks[0].first), end) &&
          equal(canonical(call("+", {end, blocks[0].second})), total))
        return call("offset_positions", {call("range", {size}), start});
    }
  return call("ordered_complement", {universe, excluded});
}
S Context::axisSequence(const S& axes) {
  // Registered views retain their spelling in canonical() so their origin
  // maps remain available. Expand sequence meaning separately for both
  // ordered equality and membership reasoning, without changing those maps.
  auto& memo = queries_.axis_sequences;
  auto comparison = [&](auto&& self, const S& x, std::size_t depth) -> S {
    tick();
    if (depth > ProofLimits::expression_depth)
      throw ProofFailure("axis comparison depth limit exceeded",
                         FailureKind::BudgetExceeded);
    if (auto it = memo.find(key(x)); it != memo.end()) return it->second;
    S result = x;
    if (x->kind == Symbol::Kind::AxisGroup) {
      auto representative = canonical(x);
      if (!equal(x, representative))
        result = self(self, representative, depth + 1);
    } else if (callIs(x, "concat")) {
      std::vector<S> parts;
      for (const auto& part : x->args)
        parts.push_back(self(self, part, depth + 1));
      result = call("concat", std::move(parts));
      if (callIs(result, "concat"))
        if (auto selection = concatSelection(result->args))
          result = self(self,
                        call("gather", {selection->base,
                                        canonical(selection->positions)}),
                        depth + 1);
    } else if (callIs(x, "gather")) {
      auto base = x->args[0], selector = canonical(x->args[1]);
      while (callIs(base, "gather")) {
        tick();
        selector = canonical(call("gather", {base->args[1], selector}));
        base = base->args[0];
      }
      base = self(self, base, depth + 1);
      result = equal(selector, canonical(call("range", {call("len", {base})})))
                   ? base
                   : call("gather", {base, selector});
      if (callIs(base, "concat") &&
          (selector->kind == Symbol::Kind::List ||
           callIs(selector, "ordered_complement")) &&
          !equal(result, base)) {
        // Context-known block lengths need not be literal in the expression
        // factory. Project an explicit selector into those blocks while
        // preserving its order (including repeated visits to a block).
        if (auto projection = projectSelection(*this, base, selector)) {
          std::vector<S> parts;
          for (const auto& part : *projection)
            parts.push_back(self(
                self, call("gather", {part.base, part.positions}), depth + 1));
          result = call("concat", std::move(parts));
          memo.emplace(key(x), result);
          return result;
        }
        std::vector<S> parts;
        bool resolved = true;
        for (const auto& position : selector->args) {
          auto part = call(
              "gather", {base, make(Symbol::Kind::List, {}, 0, 0, {position})});
          // Expand only selections that project into known producer
          // blocks. Unknown symbolic offsets retain their original view.
          if (callIs(part, "gather") && equal(part->args[0], base)) {
            resolved = false;
            break;
          }
          parts.push_back(self(self, part, depth + 1));
        }
        if (resolved) result = call("concat", std::move(parts));
      }
    } else if (callIs(x, "filter_axes")) {
      auto order = self(self, x->args[0], depth + 1);
      auto wanted = self(self, x->args[1], depth + 1);
      // Keep this comparison structural. Running the full set prover here
      // can recursively ask for the same sequence equality.
      bool retained = equal(order, wanted);
      if (callIs(wanted, "concat") || callIs(wanted, "union_axes"))
        for (const auto& part : wanted->args) retained |= equal(order, part);
      if (retained)
        result = order;
      else if (independent(order, wanted))
        result = call("empty_axes", {});
      else if (callIs(order, "concat")) {
        std::vector<S> parts;
        for (const auto& part : order->args)
          parts.push_back(
              self(self, call("filter_axes", {part, wanted}), depth + 1));
        result = call("concat", std::move(parts));
      } else
        result = call("filter_axes", {order, wanted});
    }
    memo.emplace(key(x), result);
    return result;
  };
  return comparison(comparison, axes, 0);
}
S Context::sequenceEquality(const S& a, const S& b) {
  if (a->sort == Symbol::Sort::Axes && b->sort == Symbol::Sort::Axes &&
      equal(axisSequence(a), axisSequence(b)))
    return boolean(true);
  for (const auto& [placed, interface] : {std::pair{a, b}, std::pair{b, a}}) {
    if (!callIs(placed, "place_axes") || interface->sort != Symbol::Sort::Axes)
      continue;
    auto source = placed->args[0], selector = placed->args[1],
         extra = placed->args[2];
    if (!evidence(call("distinct_axes", {interface})) ||
        !evidence(call("subset", {source, interface})) ||
        !equal(canonical(selector),
               canonical(positions(interface, source, QueryMode::Evidence))))
      continue;
    auto rest = call("ordered_complement",
                     {call("range", {call("len", {interface})}), selector});
    auto expected = select(interface, rest, QueryMode::Evidence);
    auto extra_positions = positions(interface, extra, QueryMode::Evidence);
    auto ordered_extra = orderedPositions(extra_positions);
    if (callIs(extra_positions, "gather") &&
        callIs(extra_positions->args[1], "order_positions") &&
        equal(extra_positions->args[0], extra_positions->args[1]->args[0]))
      ordered_extra =
          evidence(call("is_valid_axis_list", {extra_positions->args[0],
                                               call("len", {interface})})) ||
          (callIs(extra, "gather") &&
           evidence(call("distinct_axes", {extra->args[0]})) &&
           evidence(call("subset", {extra->args[0], interface})) &&
           mappedPositions(interface, extra->args[0]) &&
           equal(extra_positions->args[0],
                 mappedPositions(interface, extra->args[0])));
    if (equal(canonical(extra), canonical(expected)) ||
        (ordered_extra && evidence(call("same_set", {extra, expected})) &&
         evidence(call("distinct_axes", {extra})))) {
      note("axis-insertion");
      return boolean(true);
    }
  }

  // Two selections of the same distinct frame agree in order when both
  // traverse that frame monotonically and contain the same coordinates.
  if (a->sort == Symbol::Sort::Axes && b->sort == Symbol::Sort::Axes &&
      setEntails(call("same_set", {a, b}))) {
    std::set<Key> visited;
    const auto mappings = bindings_.mappingIds();
    for (auto id : mappings) {
      auto edge = bindings_.mapping(id);
      if (!visited.insert(key(edge.interface)).second) continue;
      auto pa = bindings_.lookup(edge.interface, a);
      if (!pa || !orderedPositions(pa)) continue;
      auto pb = bindings_.lookup(edge.interface, b);
      if (pb && orderedPositions(pb) &&
          evidence(call("distinct_axes", {edge.interface}))) {
        note("ordered-view-equality");
        return boolean(true);
      }
    }
  }
  if (a->sort == Symbol::Sort::Axes && b->sort == Symbol::Sort::Axes) {
    auto left = bindings_.orderedParts(a), right = bindings_.orderedParts(b);
    bool expanded =
        !equal(call("concat", left), a) || !equal(call("concat", right), b);
    if (expanded && left.size() == right.size()) {
      bool matched = true;
      for (std::size_t i = 0; i < left.size(); ++i)
        if (!equal(left[i], right[i]) &&
            !truth(canonical(call("==", {left[i], right[i]})))) {
          matched = false;
          break;
        }
      if (matched) return boolean(true);
    }
  }
  // Compare composed views without erasing their partition provenance from
  // the stored graph. Position expressions themselves already compose in the
  // expression factory.
  auto flatten = [](S x) {
    if (!callIs(x, "gather")) return x;
    auto base = x->args[0], p = x->args[1];
    while (callIs(base, "gather")) {
      p = call("gather", {base->args[1], p});
      base = base->args[0];
    }
    return call("gather", {base, p});
  };
  if (equal(flatten(a), flatten(b))) return boolean(true);
  auto sequence = [](const S& x) {
    return x->sort == Symbol::Sort::Axes || x->sort == Symbol::Sort::Extents ||
           x->sort == Symbol::Sort::Positions;
  };
  if (!sequence(a) || !sequence(b)) return call("==", {a, b});
  auto left = callIs(a, "concat") ? a->args : std::vector<S>{a};
  auto right = callIs(b, "concat") ? b->args : std::vector<S>{b};
  std::size_t start = 0;
  while (start < left.size() && start < right.size() &&
         equal(left[start], right[start]))
    ++start;
  auto le = left.size(), re = right.size();
  while (le > start && re > start && equal(left[le - 1], right[re - 1])) {
    --le;
    --re;
  }
  if (start || le != left.size()) {
    auto sequence = [&](const std::vector<S>& parts, std::size_t end) {
      if (start == end && a->sort == Symbol::Sort::Axes)
        return call("empty_axes", {});
      return call("concat",
                  std::vector<S>(parts.begin() + start, parts.begin() + end));
    };
    return call("==", {sequence(left, le), sequence(right, re)});
  }
  auto injective = [&](const S& base, const S& p, const S& q) {
    return base->sort == Symbol::Sort::Axes &&
           evidence(call("distinct_axes", {base})) &&
           evidence(call("is_valid_axis_list", {p, call("len", {base})})) &&
           evidence(call("is_valid_axis_list", {q, call("len", {base})}));
  };
  if (callIs(a, "gather") && callIs(b, "gather") &&
      equal(a->args[0], b->args[0]) &&
      injective(a->args[0], a->args[1], b->args[1]))
    return call("==", {a->args[1], b->args[1]});
  if (callIs(a, "gather") && equal(a->args[0], b) &&
      injective(b, a->args[1], call("range", {call("len", {b})})))
    return call("==", {a->args[1], call("range", {call("len", {b})})});
  if (callIs(b, "gather") && equal(b->args[0], a) &&
      injective(a, b->args[1], call("range", {call("len", {a})})))
    return call("==", {call("range", {call("len", {a})}), b->args[1]});
  return call("==", {a, b});
}
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
  if (found != facts_.representatives.end()) {
    auto result = canonical(found->second);
    queries_.canonical[original] = result;
    return result;
  }
  // Cardinality zero determines an ordered coordinate sequence completely.
  // Do this before preserving registered views: a view's selectors or domain
  // may have become empty after source metadata equalities were learned.
  if (s->sort == Symbol::Sort::Axes && !callIs(s, "empty_axes") &&
      equal(canonical(call("len", {s})), integer(0))) {
    auto empty = call("empty_axes", {});
    queries_.canonical[original] = empty;
    return empty;
  }
  // Established coordinate views are graph nodes. Preserve their origin
  // maps instead of re-expanding their expression spelling after scalar
  // metadata aliases change. Scalar selectors still normalize independently.
  if (s->sort == Symbol::Sort::Axes && bindings_.isView(s)) {
    queries_.canonical[original] = s;
    return s;
  }
  std::vector<S> args;
  for (const auto& a : s->args) args.push_back(canonical(a));
  S out = s;
  if (s->kind == Symbol::Kind::Call) {
    if (s->name == "slice" && equal(args[1], integer(0)) &&
        equal(args[2], canonical(call("len", {args[0]}))))
      out = args[0];
    else if (s->name == "index" && callIs(args[0], "concat") &&
             args[0]->sort != Symbol::Sort::Axes) {
      out = call("index", args);
      auto position = metadataInteger(args[1]);
      if (position && *position >= 0 &&
          *position <= std::numeric_limits<std::size_t>::max())
        for (const auto& part : args[0]->args) {
          auto length = metadataInteger(canonical(call("len", {part})));
          if (!length || *length < 0) break;
          if (*position < *length) {
            out = call("index", {part, integer(std::size_t(*position))});
            break;
          }
          *position -= *length;
        }
    } else if (s->name == "index" &&
               (callIs(args[0], "extent_add") ||
                callIs(args[0], "extent_sub") || callIs(args[0], "extent_mul")))
      out = call(callIs(args[0], "extent_add")   ? "+"
                 : callIs(args[0], "extent_sub") ? "-"
                                                 : "*",
                 {call("index", {args[0]->args[0], args[1]}),
                  call("index", {args[0]->args[1], args[1]})});
    else if (s->name == "index" && (callIs(args[0], "offset_extents") ||
                                    callIs(args[0], "scale_extents")))
      out =
          call(callIs(args[0], "offset_extents") ? "+" : "*",
               {call("index", {args[0]->args[0], args[1]}), args[0]->args[1]});
    else if (s->name == "index" && callIs(args[0], "gather") &&
             evidence(
                 call("is_valid_axis_list",
                      {args[0]->args[1], call("len", {args[0]->args[0]})})))
      out = call("index", {args[0]->args[0],
                           call("index", {args[0]->args[1], args[1]})});
    else if (s->name == "index" && callIs(args[0], "slice") &&
             evidence(call("<=", {integer(0), args[1]})) &&
             evidence(call("<", {args[1], call("-", {args[0]->args[2],
                                                     args[0]->args[1]})})))
      out = call("index",
                 {args[0]->args[0], call("+", {args[0]->args[1], args[1]})});
    else if (s->name == "ordered_complement" &&
             evidence(call("is_disjoint", {args[0], args[1]})))
      out = args[0];
    else if (s->name == "ordered_complement")
      out = complementBlocks(args[0], args[1]);
    else if (s->name == "==")
      out = sequenceEquality(args[0], args[1]);
    else if (s->name == "gather" && args[0]->kind == Symbol::Kind::List &&
             equal(args[0], call("range", {call("len", {args[0]})})) &&
             evidence(
                 call("is_valid_axis_list", {args[1], call("len", {args[0]})})))
      out = args[1];
    else if (s->name == "gather" && args[1]->kind == Symbol::Kind::List &&
             equal(args[1], canonical(call("range", {call("len", {args[0]})}))))
      out = args[0];
    else if (s->name == "gather" && callIs(args[1], "range") &&
             evidence(call("==", {args[1]->args[0], call("len", {args[0]})})))
      out = args[0];
    else if (s->name == "place_axes" &&
             equal(args[1], canonical(call("range", {call("len", {args[0]})}))))
      out = call("concat", {args[0], args[2]});
    else if (s->name == "concat" && args.size() > 1) {
      out = call("concat", args);
      if (auto block = positionBlock(out, true))
        out = call("offset_positions",
                   {call("range", {block->second}), block->first});
    } else if (s->name == "order_positions" &&
               (orderedPositions(args[0]) || positionBlock(args[0], true)))
      out = call("range", {call("len", {args[0]})});
    else if (s->name == "gather")
      out = selectKnown(args[0], args[1], QueryMode::Evidence);
    else if (s->name == "difference_axes" && callIs(args[1], "gather") &&
             args[1]->args[1]->kind == Symbol::Kind::List &&
             equal(args[0], args[1]->args[0]) &&
             evidence(call("distinct_axes", {args[0]})) &&
             evidence(call("is_valid_axis_list",
                           {args[1]->args[1], call("len", {args[0]})})))
      out = call(
          "gather",
          {args[0], canonical(call("ordered_complement",
                                   {call("range", {call("len", {args[0]})}),
                                    args[1]->args[1]}))});
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
    if (callIs(out, "i64") || callIs(out, "u64")) out = out->args.at(0);
    if (callIs(out, "gather") || callIs(out, "sequence_map") ||
        callIs(out, "all") || callIs(out, "any"))
      out = simplifyMetadataSequence(*this, out);
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
  if (out->kind == Symbol::Kind::List)
    out = simplifyMetadataSequence(*this, out);
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
  if (callIs(x, "place_axes"))
    return support(call("concat", {x->args[0], x->args[2]}), mode);
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
  if (callIs(x, "gather") && callIs(x->args[0], "concat") &&
      x->args[1]->kind == Symbol::Kind::List)
    if (auto projection = projectSelection(*this, x->args[0], x->args[1])) {
      std::vector<S> parts;
      for (const auto& part : *projection)
        parts.push_back(call("gather", {part.base, part.positions}));
      auto expanded = call("concat", std::move(parts));
      if (!equal(expanded, x)) return support(expanded, mode);
    }
  if (callIs(x, "concat") || callIs(x, "union_axes")) {
    std::vector<S> out;
    for (const auto& a : x->args) out.push_back(support(a, mode));
    return call("union_axes", std::move(out));
  }
  if (callIs(x, "gather")) return simplifySet(x);
  return x;
}
S Context::intersect(const S& a, const S& b, QueryMode mode) {
  tick();
  auto id = "intersection_axes:" + std::to_string(static_cast<int>(mode)) +
            ":" + key(a) + ":" + key(b);
  if (!queries_.active_set_operations.insert(id).second)
    return call("intersection_axes", {a, b});
  ActiveKey active{queries_.active_set_operations, id};
  auto x = support(a, mode), y = support(b, mode);
  if (equal(x, y)) return x;
  if (independent(x, y)) return call("empty_axes", {});
  if (partitionDisjoint(x, y)) return call("empty_axes", {});
  if (setEntails(call("disjoint", {x, y}))) return call("empty_axes", {});
  if (setEntails(call("subset", {x, y}))) return x;
  if (setEntails(call("subset", {y, x}))) return y;
  if (callIs(x, "union_axes") || callIs(x, "concat")) {
    std::vector<S> out;
    for (const auto& p : x->args) out.push_back(intersect(p, y, mode));
    return call("union_axes", std::move(out));
  }
  if (callIs(y, "union_axes") || callIs(y, "concat")) {
    std::vector<S> out;
    for (const auto& p : y->args) out.push_back(intersect(x, p, mode));
    return call("union_axes", std::move(out));
  }
  if (!elementary(x) || !elementary(y)) {
    if (callIs(x, "intersection_axes") &&
        (equal(x->args[0], y) || equal(x->args[1], y)))
      return x;
    if (callIs(y, "intersection_axes") &&
        (equal(y->args[0], x) || equal(y->args[1], x)))
      return y;
    if (symbolLess(y, x)) std::swap(x, y);
    return simplifySet(call("intersection_axes", {x, y}));
  }
  return simplifySet(call("intersection_axes", {x, y}));
}
S Context::difference(const S& a, const S& b, QueryMode mode) {
  tick();
  auto id = "difference_axes:" + std::to_string(static_cast<int>(mode)) + ":" +
            key(a) + ":" + key(b);
  if (!queries_.active_set_operations.insert(id).second)
    return call("difference_axes", {a, b});
  ActiveKey active{queries_.active_set_operations, id};
  auto x = support(a, mode), y = support(b, mode);
  if (equal(x, y)) return call("empty_axes", {});
  if (independent(x, y)) return x;
  if (partitionDisjoint(x, y)) return x;
  if (setEntails(call("disjoint", {x, y}))) return x;
  if (setEntails(call("subset", {x, y}))) return call("empty_axes", {});
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
  return simplifySet(call("difference_axes", {x, y}));
}
S Context::filter(const S& order, const S& wanted, QueryMode mode) {
  tick();
  auto w = support(wanted, mode), o = canonical(order);
  if (callIs(o, "filter_axes"))
    return filter(o->args[0], intersect(o->args[1], w, mode), mode);
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
  if (callIs(o, "gather") && equal(support(o->args[0], mode), w)) return o;
  auto overlap = intersect(o, w, mode);
  if (callIs(overlap, "empty_axes")) return overlap;
  if (equal(overlap, support(o, mode))) return o;
  w = overlap;
  // A set union is not an ordered indexing base. When its components are
  // disjoint, their concatenation is a valid coordinate sequence to reorder.
  if (callIs(w, "union_axes")) {
    auto sequence = call("concat", w->args);
    if (query(call("distinct_axes", {sequence}), mode)) w = sequence;
  }
  auto p = positions(o, w, mode);
  if (callIs(p, "ordered_complement") || callIs(p, "range"))
    return select(o, p, mode);
  if (!callIs(p, "positions") && query(call("distinct_axes", {w}), mode) &&
      query(call("is_valid_axis_list", {p, call("len", {o})}), mode))
    return select(w, call("order_positions", {p}), mode);
  return call("filter_axes", {o, w});
}
S Context::extents(const S& axes) {
  tick();
  if (callIs(axes, "filter_axes")) {
    auto group = axes->args[1];
    const auto& order = axes->args[0];
    if (callIs(order, "gather") &&
        equal(canonical(order->args[0]), canonical(group)))
      return extents(order);
    if (callIs(group, "difference_axes") &&
        equal(canonical(group->args[0]), canonical(order)) &&
        callIs(group->args[1], "gather") &&
        equal(canonical(group->args[1]->args[0]), canonical(order))) {
      auto removed = group->args[1]->args[1];
      if (evidence(call("distinct_axes", {order})) &&
          evidence(call("is_valid_axis_list", {removed, call("len", {order})})))
        return canonical(
            call("gather",
                 {extents(order),
                  call("ordered_complement",
                       {call("range", {call("len", {order})}), removed})}));
    }
    if (callIs(group, "union_axes")) group = call("concat", group->args);
    // Reuse coordinate maps rather than expand whole-tensor shape gathers.
    // Restrict this fast path to direct coordinate groups and their selections.
    auto simple = [&](auto&& self, const S& x) -> bool {
      if (callIs(x, "empty_axes")) return true;
      if (x->kind == Symbol::Kind::AxisGroup) {
        const auto& n = plan.nodes.at(x->owner);
        return n.definition && x->index < n.definition->groups.size() &&
               bool(n.definition->groups[x->index].domain);
      }
      if (callIs(x, "gather")) return self(self, x->args[0]);
      if (callIs(x, "concat")) {
        for (const auto& a : x->args)
          if (!self(self, a)) return false;
        return true;
      }
      return false;
    };
    if (simple(simple, group)) {
      auto p = mappedPositions(axes->args[0], group);
      if (p && evidence(call("distinct_axes", {group})) &&
          evidence(call("subset", {group, axes->args[0]})))
        return canonical(
            call("gather", {extents(group), call("order_positions", {p})}));
    }
  }
  for (auto it = selected_extents_.rbegin(); it != selected_extents_.rend();
       ++it) {
    if (equal(axes, it->first)) return it->second;
    if (callIs(axes, "filter_axes") && callIs(it->first, "filter_axes") &&
        equal(axes->args[0], it->first->args[0]) &&
        setEntails(call("same_set", {axes, it->first})))
      return it->second;
  }
  if (axes->kind == Symbol::Kind::AxisGroup) return axes->args[0];
  if (callIs(axes, "place_axes")) {
    auto left = extents(axes->args[0]), right = extents(axes->args[2]);
    auto rest = call("ordered_complement",
                     {call("range", {call("len", {axes})}), axes->args[1]});
    return call("gather",
                {call("concat", {left, right}),
                 call("inverse", {call("concat", {axes->args[1], rest})})});
  }
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
}  // namespace tepl::core::semantics
