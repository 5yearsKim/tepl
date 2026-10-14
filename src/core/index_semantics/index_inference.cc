#include "src/core/index_semantics/index_inference.h"

namespace tepl::core::index_semantics {
namespace {
struct Coefficient {
  I base;
  S weights;
  std::int64_t scalar = 1;
};
Coefficient peel(Context& c, I expression) {
  Coefficient out{expression};
  while (out.base->kind == IndexExpr::Kind::Scale ||
         out.base->kind == IndexExpr::Kind::Weighted) {
    c.tick();
    if (out.base->kind == IndexExpr::Kind::Scale) {
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
  if (a->kind != b->kind || a->args.size() != b->args.size()) return false;
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
bool match(Context& c, const I& pattern, const I& source,
           IndexAssignments& assignments) {
  c.tick();
  if (!sameBasis(c, pattern, source)) return false;
  auto a = peel(c, pattern), b = peel(c, source);
  if (a.weights || b.weights || a.scalar != b.scalar) {
    if (a.weights && a.weights->kind == Symbol::Kind::Hole && a.scalar == 1) {
      auto value = coefficientValue(c, b);
      if (containsHole(value))
        throw ProofFailure(
            "coefficient constructor depends on unresolved RHS metadata");
      c.validateSource(value);
      auto id = std::pair(a.weights->owner, a.weights->index);
      auto [at, inserted] = assignments.emplace(id, value);
      if (!inserted && !equalExtentArithmetic(c, at->second, value))
        throw ProofFailure("conflicting calculated-read coefficient equations",
                           FailureKind::InvalidCandidate);
    } else {
      auto left = coefficientValue(c, a), right = coefficientValue(c, b);
      if (containsHole(left))
        throw ProofFailure(
            "calculated-read coefficient equation is outside direct matching "
            "(requires inverse or coupled solving)");
      if (!equalExtentArithmetic(c, left, right)) return false;
    }
  }
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
std::optional<IndexAssignments> matchIndexCoefficients(Context& c,
                                                       const I& pattern,
                                                       const I& source) {
  IndexAssignments out;
  if (!match(c, pattern, source, out)) return std::nullopt;
  return out;
}
}  // namespace tepl::core::index_semantics
