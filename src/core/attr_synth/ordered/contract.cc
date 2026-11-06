#include "src/core/attr_synth/ordered/contract.h"

#include <functional>
#include <set>
#include <stdexcept>

#include "src/inference/evaluate.h"
#include "src/inference/extensions.h"

namespace tepl::core::attr_synth::ordered {
using namespace semantics;
namespace {
std::optional<Contract> recognizeDefinition(const Definition& d, bool extensions) {
  auto c = extensions ? axis::recognizeCoreShape(d) : axis::recognizeOrdered(d);
  if (!c || !d.value) return {};
  auto read = [](const Value& v, std::size_t i) {
    return v.kind == Value::Kind::Operand && v.index == i && v.args.empty();
  };
  if (c->kind == Contract::Kind::Negation) {
    const auto& v = *d.value;
    if (v.kind != Value::Kind::Product || v.args.size() != 2) return {};
    for (int side = 0; side < 2; ++side)
      if (v.args[side]->kind == Value::Kind::Const && v.args[side]->constant == -1 &&
          v.args[side]->args.empty() && read(*v.args[1-side], 0)) return c;
    return {};
  }
  if (c->kind == Contract::Kind::Reverse) {
    const auto& v = *d.value;
    if (v.kind != Value::Kind::Read || v.index != 0 || !v.args.empty() ||
        !v.access || v.access->kind != IndexValue::Kind::Reverse ||
        v.access->args.size() != 1 || !v.access->parameter ||
        v.access->parameter->kind != Meta::Kind::Attribute ||
        v.access->parameter->index != 0 ||
        v.access->parameter->type != axis::Type::Positions) return {};
    auto coords = v.access->args[0];
    return coords && coords->kind == IndexValue::Kind::Coordinates &&
        coords->group == 0 && coords->args.empty() && !coords->parameter ? c : std::nullopt;
  }
  if (c->kind == Contract::Kind::Permutation || c->kind == Contract::Kind::Broadcast)
    return read(*d.value, 0) ? c : std::nullopt;
  if (c->kind == Contract::Kind::Addition) {
    if (extensions && d.value->kind == Value::Kind::Product) c->kind = Contract::Kind::Multiplication;
    return (d.value->kind == Value::Kind::Add ||
            (extensions && d.value->kind == Value::Kind::Product)) && d.value->args.size() == 2 &&
                   ((read(*d.value->args[0], 0) &&
                     read(*d.value->args[1], 1)) ||
                    (read(*d.value->args[0], 1) && read(*d.value->args[1], 0)))
               ? c
               : std::nullopt;
  }
  if (d.value->kind != Value::Kind::Reduce || d.value->args.size() != 1 ||
      d.value->index != c->contract_group)
    return {};
  if (c->kind == Contract::Kind::Summation || c->kind == Contract::Kind::UnitRemoval)
    return read(*d.value->args[0], 0) ? c : std::nullopt;
  const auto& product = *d.value->args[0];
  if (product.kind != Value::Kind::Product || product.args.size() != 2 ||
      !((read(*product.args[0], 0) && read(*product.args[1], 1)) ||
        (read(*product.args[0], 1) && read(*product.args[1], 0))))
    return {};
  return c;
}

}  // namespace
std::optional<Contract> recognize(const Definition& d) { return recognizeDefinition(d, false); }

Admission admit(const Plan& p, inference::Mode mode, bool extensions) {
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
      if (!n.inputs.empty()) return fail("ordered capture has operands");
      (rhs ? target_captures : source_captures).insert(n.capture->value);
      return true;
    }
    if (!n.definition || n.literal)
      return fail(
          "ordered trees require contraction/permutation/summation "
          "definitions");
    auto c = recognizeDefinition(*n.definition, extensions);
    if (!c) return fail("operation is outside ordered axis/value contracts");
    const bool addition = c->kind == Contract::Kind::Addition || c->kind == Contract::Kind::Multiplication;
    const bool attribute_free = addition || c->kind == Contract::Kind::Negation;
    if (!attribute_free && !n.descriptor)
      return fail("ordered operation requires an axis descriptor");
    if (attribute_free && n.descriptor)
      return fail("ordered addition must be attribute-free");
    if (n.inputs.size() !=
        ((c->kind == Contract::Kind::Contraction || addition) ? 2u : 1u))
      return fail("ordered operation has wrong operand count");
    if (attribute_free) {
      // Components, rather than addition nodes, own independent descriptors.
    } else if (rhs && mode == inference::Mode::Synthesize) {
      if ((!extensions && !p.attributes(n).allAuto()) ||
          !target_descriptors.insert(n.descriptor->value).second)
        return fail(
            "ordered RHS descriptors must be independent automatic fields");
    } else if (rhs) {
      if (p.needsSynthesis(n))
        return fail(
            "mixed explicit and automatic RHS attributes are unsupported");
    } else {
      if (!p.isCaptured(n))
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
  if (!out.admitted()) return out;
  out.program.mode = mode;
  auto schedule = [&](std::size_t root) {
    std::vector<inference::Instruction> code;
    std::function<std::size_t(std::size_t)> emit = [&](std::size_t id) {
      const auto& n = p.nodes.at(id);
      inference::Instruction i;
      i.node = id;
      std::set<std::size_t> captures;
      for (auto child : n.inputs) {
        auto input = emit(child);
        i.inputs.push_back(input);
        captures.insert(code[input].captures.begin(),
                        code[input].captures.end());
      }
      if (n.capture) {
        i.capture = n.capture->value;
        captures.insert(*i.capture);
      } else {
        if (n.descriptor) i.descriptor = n.descriptor->value;
        if (n.operation) i.operation = n.operation->value;
        const auto& c = out.operations.at(id);
        switch (c.kind) {
          case Contract::Kind::Permutation:
            i.kind = inference::Kind::Permutation;
            i.fields = {c.permutation};
            break;
          case Contract::Kind::UnitRemoval:
            i.kind = inference::Kind::UnitRemoval; i.fields = {0}; break;
          case Contract::Kind::Summation:
            i.kind = inference::Kind::Summation;
            i.fields = {c.dimensions};
            break;
          case Contract::Kind::Contraction:
            i.kind = inference::Kind::Contraction;
            i.fields = {c.left_batch, c.right_batch, c.left_contract,
                        c.right_contract};
            break;
          case Contract::Kind::Addition:
            i.kind = inference::Kind::Addition;
            break;
          case Contract::Kind::Multiplication:
            i.kind = inference::Kind::Multiplication;
            break;
          case Contract::Kind::Negation:
            i.kind = inference::Kind::Negation; break;
          case Contract::Kind::Reverse:
            i.kind = inference::Kind::Reverse; i.fields = {0};
            break;
          case Contract::Kind::Broadcast:
            i.kind = inference::Kind::Broadcast; i.fields = {0, 1};
            break;
        }
      }
      i.captures.assign(captures.begin(), captures.end());
      code.push_back(std::move(i));
      return code.size() - 1;
    };
    emit(root);
    return code;
  };
  out.program.source = schedule(p.lhs);
  out.program.replacement = schedule(p.rhs);
  try {
    out.program.components = inference::decompose(out.program);
    inference::validate(out.program);
    if (extensions && !inference::hasExtensions(out.program))
      out.reason = "supplied fields remain outside canonical synthesis admission";
  } catch (const std::invalid_argument& e) {
    out.reason = e.what();
  }
  return out;
}

Plan componentPlan(const Plan& original, const inference::Component& pair) {
  auto plan = original;
  auto select = [&](const std::vector<inference::Instruction>& code) {
    for (const auto& i : code) {
      auto& node = plan.nodes.at(i.node);
      node.inputs.clear();
      for (auto input : i.inputs) node.inputs.push_back(code.at(input).node);
    }
    return code.back().node;
  };
  plan.lhs = select(pair.source);
  plan.rhs = select(pair.replacement);
  return plan;
}
}  // namespace tepl::core::attr_synth::ordered
