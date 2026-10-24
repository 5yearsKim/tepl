#include "src/core/attr_synth/ordered/contract.h"

#include <functional>
#include <set>

namespace tepl::core::attr_synth::ordered {
using namespace semantics;
std::optional<Contract> recognize(const Definition& d) {
  auto c = axis::recognizeOrdered(d);
  if (!c || !d.value) return {};
  auto read = [](const Value& v, std::size_t i) {
    return v.kind == Value::Kind::Operand && v.index == i && v.args.empty();
  };
  if (c->kind == Contract::Kind::Permutation)
    return read(*d.value, 0) ? c : std::nullopt;
  if (d.value->kind != Value::Kind::Reduce || d.value->args.size() != 1 ||
      d.value->index != c->contract_group)
    return {};
  const auto& product = *d.value->args[0];
  if (product.kind != Value::Kind::Product || product.args.size() != 2 ||
      !((read(*product.args[0], 0) && read(*product.args[1], 1)) ||
        (read(*product.args[0], 1) && read(*product.args[1], 0))))
    return {};
  return c;
}

Admission admit(const Plan& p, inference::Mode mode) {
  Admission out;
  if (!p.algebraic) {
    out.reason = "ordered contractions require ideal algebraic semantics";
    return out;
  }
  std::set<std::size_t> source_captures, target_captures;
  std::set<std::size_t> source_descriptors, target_descriptors;
  auto fail = [&](const char* reason) {
    out.reason = reason;
    return false;
  };
  std::function<bool(std::size_t, bool)> visit = [&](std::size_t id, bool rhs) {
    if (id >= p.nodes.size()) return fail("invalid ordered tree node");
    const auto& n = p.nodes[id];
    if (n.capture) {
      if (!n.inputs.empty() || !(rhs ? target_captures : source_captures)
                                    .insert(n.capture->value)
                                    .second)
        return fail("ordered trees require each capture exactly once");
      return true;
    }
    if (!n.definition || !n.descriptor || n.literal)
      return fail("ordered trees require contraction/permutation definitions");
    auto c = recognize(*n.definition);
    if (!c) return fail("operation is outside ordered axis/value contracts");
    if (n.inputs.size() != (c->kind == Contract::Kind::Permutation ? 1u : 2u))
      return fail("ordered operation has wrong operand count");
    if (rhs && mode == inference::Mode::Synthesize) {
      if (!n.automatic || p.attributes.contains(n.descriptor->value) ||
          !target_descriptors.insert(n.descriptor->value).second)
        return fail(
            "ordered RHS descriptors must be independent automatic fields");
    } else if (rhs) {
      if (n.automatic)
        return fail(
            "mixed explicit and automatic RHS attributes are unsupported");
    } else {
      if (n.automatic || p.attributes.contains(n.descriptor->value))
        return fail("ordered source attributes must be captured metadata");
      source_descriptors.insert(n.descriptor->value);
    }
    out.operations.emplace(id, *c);
    for (auto child : n.inputs) {
      if (child >= id) return fail("ordered trees must be producer-first");
      if (!visit(child, rhs)) return false;
    }
    return true;
  };
  if (!visit(p.lhs, false) || !visit(p.rhs, true)) return out;
  if (source_captures != target_captures)
    fail("ordered trees must use the same captures");
  for (auto id : target_descriptors)
    if (source_descriptors.contains(id))
      fail("RHS descriptors cannot share source attributes");
  return out;
}
}  // namespace tepl::core::attr_synth::ordered
