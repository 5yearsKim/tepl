#include "src/core/semantics/symbolic/index_expr.h"

#include <algorithm>

namespace tepl::core::semantics {
namespace {
I node(IndexExpr::Kind kind, S axes, std::vector<I> args = {},
       std::int64_t coefficient = 1, S dimensions = {}) {
  return std::make_shared<IndexExpr>(
      IndexExpr{kind, axes, args, coefficient, dimensions});
}
std::int64_t checkedAdd(std::int64_t a, std::int64_t b) {
  std::int64_t result;
  if (__builtin_add_overflow(a, b, &result))
    throw ProofFailure("index coefficient overflow");
  return result;
}
Key identity(const I& e) {
  Key out = std::to_string(int(e->kind)) + ":";
  if (e->axes) out += key(e->axes);
  if (e->dimensions) out += ":reverse=" + key(e->dimensions);
  out += ":" + std::to_string(e->coefficient);
  for (const auto& a : e->args) out += "{" + identity(a) + "}";
  return out;
}
bool indexLess(const I& a, const I& b) {
  if (a->kind != b->kind) return a->kind < b->kind;
  if (a->coefficient != b->coefficient) return a->coefficient < b->coefficient;
  if (a->axes && b->axes && !equal(a->axes, b->axes))
    return symbolLess(a->axes, b->axes);
  if (a->dimensions && b->dimensions && !equal(a->dimensions, b->dimensions))
    return symbolLess(a->dimensions, b->dimensions);
  return std::lexicographical_compare(a->args.begin(), a->args.end(),
                                      b->args.begin(), b->args.end(),
                                      indexLess);
}
S signedInteger(std::int64_t n) {
  return make(Symbol::Kind::Integer, std::to_string(n));
}
// A partial reflection preserves the box [0,extent-1]. Keep this conservative
// bound only after proving that the incoming tuple lies in that box. Empty
// axes have the empty interval [0,-1], which is also preserved. More general
// signed affine interval propagation is deliberately outside this fragment.
std::pair<S, S> reverseBounds(Context& c, const I& e) {
  auto zero = call("extent_sub", {e->axes, e->axes});
  auto last = call("offset_extents", {e->axes, signedInteger(-1)});
  auto lower = indexLowerBound(c, e->args[0]);
  auto upper = indexUpperBound(c, e->args[0]);
  bool exact = equalExtentArithmetic(c, lower, zero) &&
               equalExtentArithmetic(c, upper, last);
  if (!exact && !(c.entails(call("is_valid_extents", {lower})) &&
                  extentBoundArithmetic(c, upper, last, QueryMode::Entailment)))
    throw ProofFailure(
        "reverse coordinates are not established within their "
        "original extent domain");
  return {zero, last};
}
}  // namespace
I coordinates(S axes) { return node(IndexExpr::Kind::Coordinates, axes); }
I indexConcat(std::vector<I> args) {
  std::vector<I> flat;
  for (const auto& a : args) {
    if (a->kind == IndexExpr::Kind::Concat)
      flat.insert(flat.end(), a->args.begin(), a->args.end());
    else if (a->kind != IndexExpr::Kind::Coordinates ||
             !callIs(a->axes, "empty_axes"))
      flat.push_back(a);
  }
  if (flat.empty()) return coordinates(call("empty_axes", {}));
  if (flat.size() == 1) return flat[0];
  // Adjacent identity blocks are an ordinary axis sequence.
  bool direct = std::all_of(flat.begin(), flat.end(), [](const I& a) {
    return a->kind == IndexExpr::Kind::Coordinates;
  });
  if (direct) {
    std::vector<S> axes;
    for (const auto& a : flat) axes.push_back(a->axes);
    return coordinates(call("concat", axes));
  }
  return node(IndexExpr::Kind::Concat, {}, flat);
}
I indexAdd(std::vector<I> args) {
  if (args.empty())
    throw ProofFailure("index addition requires coordinate tuples");
  S offsets;
  std::map<Key, std::pair<I, std::int64_t>> terms;
  auto collect = [&](auto&& self, const I& e, std::int64_t factor) -> void {
    if (e->kind == IndexExpr::Kind::Offset) {
      auto part = factor == 1
                      ? e->axes
                      : call("scale_extents", {e->axes, signedInteger(factor)});
      offsets = offsets ? call("extent_add", {offsets, part}) : part;
      self(self, e->args[0], factor);
      return;
    }
    if (e->kind == IndexExpr::Kind::Add) {
      for (const auto& a : e->args) self(self, a, factor);
      return;
    }
    if (e->kind == IndexExpr::Kind::Scale) {
      if (__builtin_mul_overflow(factor, e->coefficient, &factor))
        throw ProofFailure("index coefficient overflow");
      self(self, e->args[0], factor);
      return;
    }
    auto& entry = terms[identity(e)];
    entry.first = e;
    entry.second = checkedAdd(entry.second, factor);
  };
  for (const auto& a : args) collect(collect, a, 1);
  std::vector<I> normalized;
  for (const auto& [id, entry] : terms) {
    auto [term, coefficient] = entry;
    normalized.push_back(coefficient == 1 ? term
                                          : node(IndexExpr::Kind::Scale, {},
                                                 {term}, coefficient));
  }
  std::sort(normalized.begin(), normalized.end(), indexLess);
  auto out = normalized.size() == 1
                 ? normalized[0]
                 : node(IndexExpr::Kind::Add, {}, normalized);
  return offsets ? node(IndexExpr::Kind::Offset, offsets, {out}) : out;
}
I indexOffset(Context& c, const S& raw, const I& e) {
  c.tick();
  auto offsets = c.canonical(raw);
  if (offsets->kind == Symbol::Kind::List &&
      std::all_of(offsets->args.begin(), offsets->args.end(), [](const S& x) {
        return x->kind == Symbol::Kind::Integer && x->name == "0";
      }))
    return e;
  if (e->kind == IndexExpr::Kind::Offset)
    return indexOffset(c, call("extent_add", {offsets, e->axes}), e->args[0]);
  return node(IndexExpr::Kind::Offset, offsets, {e});
}
I indexScale(Context& c, const S& raw, const I& e) {
  c.tick();
  auto factors = c.canonical(raw);
  if (factors->kind == Symbol::Kind::List &&
      std::all_of(factors->args.begin(), factors->args.end(), [](const S& x) {
        return x->kind == Symbol::Kind::Integer && x->name == "1";
      }))
    return e;
  if (e->kind == IndexExpr::Kind::Offset)
    return indexOffset(c, call("extent_mul", {factors, e->axes}),
                       indexScale(c, factors, e->args[0]));
  if (e->kind == IndexExpr::Kind::Weighted)
    return indexScale(c, call("extent_mul", {factors, e->axes}), e->args[0]);
  if (e->kind == IndexExpr::Kind::Scale)
    return node(e->kind, {}, {indexScale(c, factors, e->args[0])},
                e->coefficient);
  if (e->kind == IndexExpr::Kind::Add) {
    std::vector<I> args;
    for (const auto& a : e->args) args.push_back(indexScale(c, factors, a));
    return indexAdd(args);
  }
  return node(IndexExpr::Kind::Weighted, factors, {e});
}
I indexReverse(Context& c, const I& e, const S& raw_extents,
               const S& raw_dimensions) {
  c.tick();
  auto extents = c.canonical(raw_extents);
  auto dimensions = c.canonical(raw_dimensions);
  if (equal(c.canonical(call("len", {dimensions})), integer(0))) return e;
  if (equal(c.canonical(indexLength(e)), integer(0))) return e;
  if (e->kind == IndexExpr::Kind::Reverse &&
      equalExtentArithmetic(c, extents, e->axes) &&
      c.entails(call("==", {dimensions, e->dimensions}))) {
    c.note("index-reverse-cancellation");
    return e->args[0];
  }
  // An extent-one coordinate is necessarily zero. Do not apply this law to
  // arbitrary calculated expressions just because their reflection extent is 1.
  if (e->kind == IndexExpr::Kind::Coordinates &&
      equalExtentArithmetic(c, extents, c.extents(e->axes))) {
    auto selected = c.canonical(call("gather", {extents, dimensions}));
    auto ones = call("offset_extents",
                     {call("extent_sub", {selected, selected}), integer(1)});
    bool literal_ones =
        selected->kind == Symbol::Kind::List &&
        std::all_of(selected->args.begin(), selected->args.end(),
                    [](const S& x) { return equal(x, integer(1)); });
    if (literal_ones || equalExtentArithmetic(c, selected, ones)) {
      c.note("index-reverse-singleton");
      return e;
    }
  }
  return node(IndexExpr::Kind::Reverse, extents, {e}, 1, dimensions);
}
S indexLength(const I& e) {
  if (e->kind == IndexExpr::Kind::Coordinates) return call("len", {e->axes});
  if (e->kind == IndexExpr::Kind::Select) return call("len", {e->axes});
  if (e->kind != IndexExpr::Kind::Concat) return indexLength(e->args[0]);
  S n = integer(0);
  for (const auto& a : e->args) n = call("+", {n, indexLength(a)});
  return n;
}
S indexDependencies(const I& e) {
  if (e->kind == IndexExpr::Kind::Coordinates) return e->axes;
  std::vector<S> axes;
  for (const auto& a : e->args) axes.push_back(indexDependencies(a));
  return call("union_axes", axes);
}
I indexSelect(Context& c, const I& e, const S& selector) {
  c.tick();
  auto p = c.canonical(selector);
  if (equal(p, c.canonical(call("range", {indexLength(e)})))) return e;
  if (e->kind == IndexExpr::Kind::Coordinates)
    return coordinates(c.select(e->axes, p));
  if (e->kind == IndexExpr::Kind::Select)
    return indexSelect(c, e->args[0],
                       c.canonical(call("gather", {e->axes, p})));
  if (e->kind == IndexExpr::Kind::Add) {
    std::vector<I> args;
    for (const auto& a : e->args) args.push_back(indexSelect(c, a, p));
    return indexAdd(args);
  }
  if (e->kind == IndexExpr::Kind::Offset)
    return indexOffset(c, call("gather", {e->axes, p}),
                       indexSelect(c, e->args[0], p));
  if (e->kind == IndexExpr::Kind::Weighted)
    return indexScale(c, call("gather", {e->axes, p}),
                      indexSelect(c, e->args[0], p));
  if (e->kind == IndexExpr::Kind::Reverse)
    return indexReverse(c, indexSelect(c, e->args[0], p),
                        call("gather", {e->axes, p}),
                        call("membership_positions", {p, e->dimensions}));
  if (e->kind == IndexExpr::Kind::Scale)
    return node(e->kind, {}, {indexSelect(c, e->args[0], p)}, e->coefficient);
  // Resolve exact block selections without expanding symbolic group members.
  S offset = integer(0);
  for (const auto& a : e->args) {
    auto block = c.canonical(
        call("offset_positions", {call("range", {indexLength(a)}), offset}));
    if (equal(p, block)) return a;
    offset = call("+", {offset, indexLength(a)});
  }
  if (callIs(p, "concat")) {
    std::vector<I> args;
    for (const auto& part : p->args) args.push_back(indexSelect(c, e, part));
    return indexConcat(args);
  }
  return node(IndexExpr::Kind::Select, p, {e});
}
I indexSubstitute(Context& c, const I& e, const S& frame,
                  const I& replacement) {
  c.tick();
  c.enableSetReasoning();
  if (e->kind == IndexExpr::Kind::Coordinates) {
    if (equal(e->axes, frame)) return replacement;
    if (auto known = c.mappedPositions(frame, e->axes))
      return indexSelect(c, replacement, known);
    if (c.entails(call("==", {e->axes, frame}))) return replacement;
    if (callIs(e->axes, "concat")) {
      std::vector<I> pieces;
      for (const auto& a : e->axes->args)
        pieces.push_back(
            indexSubstitute(c, coordinates(a), frame, replacement));
      return indexConcat(pieces);
    }
    if (callIs(e->axes, "gather")) {
      auto base =
          indexSubstitute(c, coordinates(e->axes->args[0]), frame, replacement);
      return indexSelect(c, base, e->axes->args[1]);
    }
    if (c.entails(call("subset", {e->axes, frame})))
      return indexSelect(c, replacement, c.positions(frame, e->axes));
    if (c.entails(call("disjoint", {e->axes, frame}))) return e;
    throw ProofFailure(
        "partial coordinate substitution is outside the access fragment: " +
        format(*e->axes) + " within " + format(*frame));
  }
  std::vector<I> args;
  for (const auto& a : e->args)
    args.push_back(indexSubstitute(c, a, frame, replacement));
  if (e->kind == IndexExpr::Kind::Add) return indexAdd(args);
  if (e->kind == IndexExpr::Kind::Concat) return indexConcat(args);
  if (e->kind == IndexExpr::Kind::Select)
    return indexSelect(c, args[0], e->axes);
  if (e->kind == IndexExpr::Kind::Weighted)
    return indexScale(c, e->axes, args[0]);
  if (e->kind == IndexExpr::Kind::Offset)
    return indexOffset(c, e->axes, args[0]);
  if (e->kind == IndexExpr::Kind::Reverse)
    return indexReverse(c, args[0], e->axes, e->dimensions);
  if (e->kind == IndexExpr::Kind::Scale &&
      args[0]->kind == IndexExpr::Kind::Offset)
    return indexOffset(
        c,
        call("scale_extents", {args[0]->axes, signedInteger(e->coefficient)}),
        node(e->kind, {}, {args[0]->args[0]}, e->coefficient));
  return node(e->kind, {}, args, e->coefficient);
}
I indexRename(Context& c, const I& e, const S& from, const S& to) {
  c.tick();
  if (e->kind == IndexExpr::Kind::Coordinates)
    return coordinates(c.transportInterface(from, to, e->axes));
  std::vector<I> args;
  for (const auto& a : e->args) args.push_back(indexRename(c, a, from, to));
  if (e->kind == IndexExpr::Kind::Add) return indexAdd(args);
  if (e->kind == IndexExpr::Kind::Concat) return indexConcat(args);
  if (e->kind == IndexExpr::Kind::Select)
    return indexSelect(c, args[0], e->axes);
  if (e->kind == IndexExpr::Kind::Weighted)
    return indexScale(c, e->axes, args[0]);
  if (e->kind == IndexExpr::Kind::Offset)
    return indexOffset(c, e->axes, args[0]);
  if (e->kind == IndexExpr::Kind::Reverse)
    return indexReverse(c, args[0], e->axes, e->dimensions);
  if (e->kind == IndexExpr::Kind::Scale &&
      args[0]->kind == IndexExpr::Kind::Offset)
    return indexOffset(
        c,
        call("scale_extents", {args[0]->axes, signedInteger(e->coefficient)}),
        node(e->kind, {}, {args[0]->args[0]}, e->coefficient));
  return node(e->kind, {}, args, e->coefficient);
}
S indexUpperBound(Context& c, const I& e) {
  c.tick();
  if (e->kind == IndexExpr::Kind::Coordinates)
    return call("offset_extents", {c.extents(e->axes), signedInteger(-1)});
  if (e->kind == IndexExpr::Kind::Reverse) return reverseBounds(c, e).second;
  std::vector<S> args;
  for (const auto& a : e->args) args.push_back(indexUpperBound(c, a));
  if (e->kind == IndexExpr::Kind::Concat) return call("concat", args);
  if (e->kind == IndexExpr::Kind::Select)
    return call("gather", {args[0], e->axes});
  if (e->kind == IndexExpr::Kind::Offset)
    return call("extent_add", {e->axes, args[0]});
  if (e->kind == IndexExpr::Kind::Weighted)
    return call("extent_mul", {e->axes, args[0]});
  if (e->kind == IndexExpr::Kind::Scale) {
    if (e->coefficient < 1) throw ProofFailure("nonpositive index scale");
    return call("scale_extents", {args[0], signedInteger(e->coefficient)});
  }
  S out = args[0];
  for (std::size_t i = 1; i < args.size(); ++i)
    out = call("extent_add", {out, args[i]});
  return out;
}
S indexLowerBound(Context& c, const I& e) {
  c.tick();
  if (e->kind == IndexExpr::Kind::Coordinates) {
    auto extents = c.extents(e->axes);
    return call("extent_sub", {extents, extents});
  }
  if (e->kind == IndexExpr::Kind::Reverse) return reverseBounds(c, e).first;
  std::vector<S> args;
  for (const auto& a : e->args) args.push_back(indexLowerBound(c, a));
  if (e->kind == IndexExpr::Kind::Concat) return call("concat", args);
  if (e->kind == IndexExpr::Kind::Select)
    return call("gather", {args[0], e->axes});
  if (e->kind == IndexExpr::Kind::Offset)
    return call("extent_add", {e->axes, args[0]});
  if (e->kind == IndexExpr::Kind::Weighted)
    return call("extent_mul", {e->axes, args[0]});
  if (e->kind == IndexExpr::Kind::Scale) {
    if (e->coefficient < 1) throw ProofFailure("nonpositive index scale");
    return call("scale_extents", {args[0], signedInteger(e->coefficient)});
  }
  S out = args[0];
  for (std::size_t i = 1; i < args.size(); ++i)
    out = call("extent_add", {out, args[i]});
  return out;
}
bool equalIndex(Context& c, const I& a, const I& b) {
  c.tick();
  if (a->kind != b->kind) return false;
  if (a->coefficient != b->coefficient || a->args.size() != b->args.size())
    return false;
  if (a->kind == IndexExpr::Kind::Reverse &&
      !c.entails(call("==", {a->dimensions, b->dimensions})))
    return false;
  if (a->axes && !((a->kind == IndexExpr::Kind::Weighted ||
                    a->kind == IndexExpr::Kind::Offset ||
                    a->kind == IndexExpr::Kind::Reverse)
                       ? equalExtentArithmetic(c, a->axes, b->axes)
                       : c.entails(call("==", {a->axes, b->axes}))))
    return false;
  if (a->kind == IndexExpr::Kind::Add) {
    std::vector<bool> used(b->args.size());
    for (const auto& x : a->args) {
      bool found = false;
      for (std::size_t i = 0; i < b->args.size(); ++i)
        if (!used[i] && equalIndex(c, x, b->args[i])) {
          used[i] = true;
          found = true;
          break;
        }
      if (!found) return false;
    }
    return true;
  }
  for (std::size_t i = 0; i < a->args.size(); ++i)
    if (!equalIndex(c, a->args[i], b->args[i])) return false;
  return true;
}
std::string showIndex(const I& e) {
  if (e->kind == IndexExpr::Kind::Coordinates)
    return "coords(" + format(*e->axes) + ")";
  std::string out = e->kind == IndexExpr::Kind::Add        ? "index_add("
                    : e->kind == IndexExpr::Kind::Concat   ? "index_concat("
                    : e->kind == IndexExpr::Kind::Select   ? "index_select("
                    : e->kind == IndexExpr::Kind::Offset   ? "index_offset("
                    : e->kind == IndexExpr::Kind::Weighted ? "index_weighted("
                    : e->kind == IndexExpr::Kind::Reverse  ? "index_reverse("
                                                           : "index_scale(";
  for (std::size_t i = 0; i < e->args.size(); ++i) {
    if (i) out += ", ";
    out += showIndex(e->args[i]);
  }
  if (e->kind == IndexExpr::Kind::Select ||
      e->kind == IndexExpr::Kind::Weighted ||
      e->kind == IndexExpr::Kind::Offset || e->kind == IndexExpr::Kind::Reverse)
    out += ", " + format(*e->axes);
  if (e->kind == IndexExpr::Kind::Scale)
    out += ", " + std::to_string(e->coefficient);
  if (e->kind == IndexExpr::Kind::Reverse)
    out += ", dimensions=" + format(*e->dimensions);
  return out + ")";
}
}  // namespace tepl::core::semantics
