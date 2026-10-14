#include "src/core/index_semantics/index_inference.h"

namespace tepl::core::index_semantics {
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
bool bindMetadata(Context& c, const S& pattern, const S& source,
                  IndexAssignments& assignments) {
  c.tick();
  auto left = c.canonical(pattern), right = c.canonical(source);
  if (left->kind == Symbol::Kind::Hole) {
    if (containsHole(right))
      throw ProofFailure("constructor depends on unresolved RHS metadata");
    c.validateSource(right);
    auto id = std::pair(left->owner, left->index);
    auto [old, inserted] = assignments.emplace(id, right);
    if (!inserted && !equalExtentArithmetic(c, old->second, right))
      throw ProofFailure(
          "conflicting calculated-read coefficient or offset equations",
          FailureKind::InvalidCandidate);
    return true;
  }
  // Identical selections admit the sufficient constructor of copying the
  // source vector. Partial selections need not determine every possible value.
  if (callIs(left, "gather") && callIs(right, "gather") &&
      c.entails(call("==", {left->args[1], right->args[1]})))
    return bindMetadata(c, left->args[0], right->args[0], assignments);
  if (containsHole(left))
    throw ProofFailure(
        "calculated-read equation is outside direct matching (requires inverse "
        "or coupled solving)");
  return equalExtentArithmetic(c, left, right);
}
bool match(Context& c, const I& pattern, const I& source,
           IndexAssignments& assignments) {
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
std::optional<IndexAssignments> matchIndexAttributes(Context& c,
                                                     const I& pattern,
                                                     const I& source) {
  IndexAssignments out;
  if (!match(c, pattern, source, out)) return std::nullopt;
  return out;
}
}  // namespace tepl::core::index_semantics
