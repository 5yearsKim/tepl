#include "src/rewrite/builder.h"

#include <functional>

namespace tepl::rewrite {
std::size_t build(const executable::ProgramData& p, const core::BuildExpr& e,
                  core::ConcreteGraph& g, const Bindings& b, Budget& budget,
                  std::size_t max_nodes) {
  budget.tick();
  return std::visit(
      [&](const auto& v) -> std::size_t {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, core::CaptureRef>)
          return b.captures.at(v.capture.value);
        else {
          core::ConcreteNode node;
          node.origin = e.origin;
          if constexpr (std::is_same_v<T, core::GraphLiteral>) {
            node.kind = core::ConcreteNode::Kind::kLiteral;
            node.text = v.spelling;
            node.dtype = v.dtype;
          } else {
            node.kind = core::ConcreteNode::Kind::kOperation;
            node.operation = v.operation;
            const auto& op = p.operations.at(v.operation.value);
            node.text = op.dialect + "." + op.name;
            for (const auto& a : v.operands)
              node.operands.push_back(build(p, *a, g, b, budget, max_nodes));
            if (op.attributes) {
              const auto& schema = p.attribute_schemas.at(op.attributes->value);
              if (v.descriptor) {
                const auto& attrs = b.descriptors.at(v.descriptor->value);
                if (attrs.schema != schema.id.value ||
                    attrs.fields.size() != schema.fields.size())
                  throw EvaluationError(
                      "replacement attribute schema mismatch");
                for (std::size_t i = 0; i < schema.fields.size(); ++i)
                  node.attributes.push_back(
                      attribute(attrs.fields[i], schema.fields[i]));
              } else
                for (const auto& field : schema.fields) {
                  if (field.empty_default)
                    node.attributes.push_back(
                        {core::AttributeValue::Kind::kList, {}, {}});
                  else if (field.optional)
                    node.attributes.push_back(
                        {core::AttributeValue::Kind::kNone, {}, {}});
                  else
                    throw EvaluationError("missing RHS attributes");
                }
            }
          }
          if (g.nodes.size() >= max_nodes) throw LimitError("graph node limit");
          auto id = g.nodes.size();
          g.nodes.push_back(std::move(node));
          infer(p, g, id, budget);
          return id;
        }
      },
      e.value);
}
core::ConcreteGraph replace(const core::ConcreteGraph& source,
                            std::size_t target, std::size_t replacement,
                            std::size_t old_size) {
  core::ConcreteGraph out = source;
  out.nodes.clear();
  out.bindings.clear();
  std::map<std::pair<std::size_t, bool>, std::size_t> remap;
  std::vector<bool> affected(source.nodes.size());
  for (std::size_t i = 0; i < source.nodes.size(); ++i) {
    affected[i] = i == target;
    for (auto a : source.nodes[i].operands)
      affected[i] = affected[i] || affected[a];
  }
  auto keyFor = [&](std::size_t id, bool rewrite) {
    return std::pair{id, rewrite && affected[id]};
  };
  auto copy = [&](std::size_t id, bool rewrite) {
    const auto root = keyFor(id, rewrite);
    std::vector<std::pair<std::size_t, bool>> pending{root};
    while (!pending.empty()) {
      auto key = pending.back();
      if (remap.contains(key)) {
        pending.pop_back();
        continue;
      }
      auto [current, change] = key;
      if (change && current == target) {
        auto replacementKey = keyFor(replacement, false);
        if (!remap.contains(replacementKey)) {
          pending.push_back(replacementKey);
          continue;
        }
        remap[key] = remap.at(replacementKey);
        pending.pop_back();
        continue;
      }
      bool ready = true;
      for (auto a : source.nodes[current].operands) {
        auto child = keyFor(a, change);
        if (!remap.contains(child)) {
          pending.push_back(child);
          ready = false;
        }
      }
      if (!ready) continue;
      auto n = source.nodes[current];
      for (auto& a : n.operands) a = remap.at(keyFor(a, change));
      remap[key] = out.nodes.size();
      out.nodes.push_back(std::move(n));
      pending.pop_back();
    }
    return remap.at(root);
  };
  for (std::size_t i = 0; i < old_size; ++i)
    if (source.nodes[i].kind == core::ConcreteNode::Kind::kInput) copy(i, true);
  out.root = copy(source.root, true);
  for (const auto& [name, id] : source.bindings) {
    auto it = remap.find({id, true});
    if (it == remap.end()) it = remap.find({id, false});
    if (it != remap.end()) out.bindings.emplace_back(name, it->second);
  }
  return out;
}
bool sameGraph(const core::ConcreteGraph& a, const core::ConcreteGraph& b) {
  std::set<std::pair<std::size_t, std::size_t>> seen;
  std::map<std::size_t, std::size_t> forward, backward;
  std::vector<std::pair<std::size_t, std::size_t>> pending{{a.root, b.root}};
  while (!pending.empty()) {
    auto [i, j] = pending.back();
    pending.pop_back();
    if ((forward.contains(i) && forward.at(i) != j) ||
        (backward.contains(j) && backward.at(j) != i))
      return false;
    forward[i] = j;
    backward[j] = i;
    if (!seen.insert({i, j}).second) continue;
    const auto& x = a.nodes[i];
    const auto& y = b.nodes[j];
    if (x.kind != y.kind || x.operation != y.operation || x.dtype != y.dtype ||
        x.shape != y.shape || x.operands.size() != y.operands.size() ||
        x.attributes.size() != y.attributes.size())
      return false;
    if (x.kind != core::ConcreteNode::Kind::kOperation && x.text != y.text)
      return false;
    for (std::size_t k = 0; k < x.attributes.size(); ++k)
      if (attributeValue(x.attributes[k]) != attributeValue(y.attributes[k]))
        return false;
    for (std::size_t k = 0; k < x.operands.size(); ++k)
      pending.emplace_back(x.operands[k], y.operands[k]);
  }
  return true;
}
}  // namespace tepl::rewrite
