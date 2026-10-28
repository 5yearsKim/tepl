#include "src/core/attr_synth/axis_selectors.h"

namespace tepl::core::attr_synth {
using namespace semantics;
namespace {
struct Coefficient {
  I base;
  S weights;
  std::int64_t scalar = 1;
  S offset;
};
Coefficient peel(Context& c, I expression) {
  Coefficient out{expression};
  while (out.base->kind == IndexExpr::Kind::Scale ||
         out.base->kind == IndexExpr::Kind::Weighted ||
         out.base->kind == IndexExpr::Kind::Offset) {
    c.tick();
    if (out.base->kind == IndexExpr::Kind::Offset) {
      auto part = c.canonical(out.base->axes);
      if (out.weights) part = call("extent_mul", {out.weights, part});
      if (out.scalar != 1)
        part = call("scale_extents", {part, make(Symbol::Kind::Integer,
                                                 std::to_string(out.scalar))});
      out.offset = out.offset ? call("extent_add", {out.offset, part}) : part;
    } else if (out.base->kind == IndexExpr::Kind::Scale) {
      if (__builtin_mul_overflow(out.scalar, out.base->coefficient,
                                 &out.scalar))
        throw ProofFailure("index coefficient overflow");
    } else {
      auto weights = c.canonical(out.base->axes);
      out.weights =
          out.weights ? call("extent_mul", {out.weights, weights}) : weights;
    }
    out.base = out.base->args[0];
  }
  return out;
}
// Correspondence is about coordinate identities, not their extents or spelling.
bool sameBasis(Context& c, const I& pattern, const I& source) {
  c.tick();
  auto a = peel(c, pattern).base, b = peel(c, source).base;
  // A reversal with an unknown selector can represent an unchanged access.
  // This supplies the empty selector only when the child already has the
  // source's coordinate identity; final verification checks the constructor.
  if (a->kind == IndexExpr::Kind::Reverse &&
      b->kind != IndexExpr::Kind::Reverse &&
      a->dimensions->kind == Symbol::Kind::Hole)
    return sameBasis(c, a->args[0], b);
  if (a->kind != b->kind || a->args.size() != b->args.size()) return false;
  if (a->kind == IndexExpr::Kind::Reverse && !containsHole(a->dimensions) &&
      !c.entails(call("==", {a->dimensions, b->dimensions})))
    return false;
  if (a->axes && !c.entails(call("==", {a->axes, b->axes}))) return false;
  if (a->kind == IndexExpr::Kind::Add) {
    std::vector<bool> used(b->args.size());
    for (const auto& x : a->args) {
      std::size_t found = b->args.size();
      for (std::size_t i = 0; i < b->args.size(); ++i)
        if (!used[i] && sameBasis(c, x, b->args[i])) {
          if (found != b->args.size())
            throw ProofFailure(
                "ambiguous coordinate basis in calculated-read inference");
          found = i;
        }
      if (found == b->args.size()) return false;
      used[found] = true;
    }
    return true;
  }
  for (std::size_t i = 0; i < a->args.size(); ++i)
    if (!sameBasis(c, a->args[i], b->args[i])) return false;
  return true;
}
S coefficientValue(Context& c, const Coefficient& value) {
  auto weights = value.weights;
  if (!weights) {
    // Uniform one tuple with the existing source tuple's length. This does not
    // enumerate rank or introduce a coordinate identity.
    auto domain = indexUpperBound(c, value.base);
    weights = call("offset_extents",
                   {call("extent_sub", {domain, domain}), integer(1)});
  }
  if (value.scalar != 1)
    weights = call(
        "scale_extents",
        {weights, make(Symbol::Kind::Integer, std::to_string(value.scalar))});
  return c.canonical(weights);
}
// Known coordinate parameters are checked, never synthesized here.
bool bindMetadata(Context& c, const S& pattern, const S& source,
                  AxisAssignments&) {
  if (containsHole(pattern) || containsHole(source))
    throw ProofFailure("coordinate parameters must be supplied explicitly");
  return equalExtentArithmetic(c, pattern, source);
}
bool bindReverseSelector(Context& c, const S& pattern, const S& source,
                         AxisAssignments& assignments) {
  c.tick();
  auto left = c.canonical(pattern), right = c.canonical(source);
  if (left->kind == Symbol::Kind::Hole) {
    if (containsHole(right))
      throw ProofFailure("reverse selector depends on unresolved RHS metadata");
    c.validateSource(right);
    auto id = std::pair(left->owner, left->index);
    auto [old, inserted] = assignments.emplace(id, right);
    if (!inserted && !c.entails(call("==", {old->second, right})))
      throw ProofFailure("conflicting reverse selector constructors",
                         FailureKind::InvalidCandidate);
    return true;
  }
  if (containsHole(left))
    throw ProofFailure(
        "reverse selector equation is outside direct matching (requires "
        "inverse or coupled solving)");
  return c.entails(call("==", {left, right}));
}
bool match(Context& c, const I& pattern, const I& source,
           AxisAssignments& assignments) {
  c.tick();
  if (!sameBasis(c, pattern, source)) return false;
  auto a = peel(c, pattern), b = peel(c, source);
  if (a.offset || b.offset) {
    auto zero = [&] {
      auto domain = indexUpperBound(c, b.base);
      return c.canonical(call("extent_sub", {domain, domain}));
    };
    auto left = a.offset ? c.canonical(a.offset) : zero();
    auto right = b.offset ? c.canonical(b.offset) : zero();
    if (!bindMetadata(c, left, right, assignments)) return false;
  }
  if (a.weights || b.weights || a.scalar != b.scalar) {
    if (a.scalar != 1 && a.weights && containsHole(a.weights))
      throw ProofFailure(
          "calculated-read coefficient equation is outside direct matching "
          "(requires inverse or coupled solving)");
    if (!bindMetadata(c, coefficientValue(c, a), coefficientValue(c, b),
                      assignments))
      return false;
  }
  if (a.base->kind == IndexExpr::Kind::Reverse &&
      b.base->kind != IndexExpr::Kind::Reverse) {
    if (!bindReverseSelector(c, a.base->dimensions, make(Symbol::Kind::List),
                             assignments))
      return false;
    return match(c, a.base->args[0], b.base, assignments);
  }
  if (a.base->kind == IndexExpr::Kind::Reverse &&
      !bindReverseSelector(c, a.base->dimensions, b.base->dimensions,
                           assignments))
    return false;
  if (a.base->kind == IndexExpr::Kind::Add) {
    std::vector<bool> used(b.base->args.size());
    for (const auto& x : a.base->args) {
      std::size_t found = b.base->args.size();
      for (std::size_t i = 0; i < b.base->args.size(); ++i)
        if (!used[i] && sameBasis(c, x, b.base->args[i])) {
          found = i;
          break;
        }
      if (found == b.base->args.size() ||
          !match(c, x, b.base->args[found], assignments))
        return false;
      used[found] = true;
    }
    return true;
  }
  for (std::size_t i = 0; i < a.base->args.size(); ++i)
    if (!match(c, a.base->args[i], b.base->args[i], assignments)) return false;
  return true;
}
}  // namespace
std::optional<AxisAssignments> matchAxisSelectors(Context& c, const I& pattern,
                                                  const I& source) {
  AxisAssignments out;
  if (!match(c, pattern, source, out)) return std::nullopt;
  return out;
}
}  // namespace tepl::core::attr_synth
