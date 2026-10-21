#include "src/shared/metadata/ordered.h"

namespace tepl::core::axis {
namespace {
bool named(const Meta& m, const char* name, std::size_t arity) {
  return m.kind == Meta::Kind::Call && m.name == name && m.args.size() == arity;
}
bool operand(const Meta& m, std::size_t i) {
  return m.kind == Meta::Kind::Operand && m.index == i;
}
bool all(const Meta& m, std::size_t i) {
  return named(m, "all_axes", 1) && operand(*m.args[0], i);
}
bool rank(const Meta& m, std::size_t i) {
  return named(m, "rank", 1) && operand(*m.args[0], i);
}
bool field(const Meta& m, std::size_t i) {
  return m.kind == Meta::Kind::Attribute && m.type == Type::Positions &&
         m.index == i;
}
void flatten(const Meta& m, std::vector<const Meta*>& out) {
  if (m.kind == Meta::Kind::Call && m.name == "concat") {
    for (const auto& a : m.args) flatten(*a, out);
  } else {
    out.push_back(&m);
  }
}
bool freePositions(const Meta& m, std::size_t o, std::size_t b, std::size_t k) {
  if (!named(m, "ordered_complement", 2) || !all(*m.args[0], o)) return false;
  std::vector<const Meta*> selected;
  flatten(*m.args[1], selected);
  return selected.size() == 2 &&
         ((field(*selected[0], b) && field(*selected[1], k)) ||
          (field(*selected[0], k) && field(*selected[1], b)));
}
bool permutationAssertion(const Meta& m, std::size_t p) {
  if (named(m, "is_valid_axis_list", 2))
    return field(*m.args[0], p) && rank(*m.args[1], 0);
  if (!named(m, "==", 2)) return false;
  auto length = [&](const Meta& a, const Meta& b) {
    return named(a, "len", 1) && field(*a.args[0], p) && rank(b, 0);
  };
  return length(*m.args[0], *m.args[1]) || length(*m.args[1], *m.args[0]);
}
}  // namespace

std::optional<OrderedContract> recognizeOrdered(const ShapeContract& d) {
  if (!d.output || !d.extent_fields.empty()) return {};
  if (d.operands.size() == 1 && d.fields.size() == 1 && d.groups.size() == 1 &&
      d.implicit_operands == std::set<std::size_t>{0}) {
    const auto& g = d.groups[0];
    if (g.domain || g.bindings.size() != 1 || g.bindings[0].operand != 0 ||
        !all(*g.bindings[0].positions, 0) || !named(*d.output, "permute", 2) ||
        d.output->args[0]->kind != Meta::Kind::Group ||
        d.output->args[0]->index != 0 || !field(*d.output->args[1], 0))
      return {};
    for (const auto& a : d.assertions)
      if (!permutationAssertion(*a, 0)) return {};
    return OrderedContract{OrderedContract::Kind::Permutation};
  }
  if (d.operands.size() != 2 || d.fields.size() != 4 || d.groups.size() != 4 ||
      !d.assertions.empty() ||
      d.implicit_operands != std::set<std::size_t>{0, 1})
    return {};
  OrderedContract c{OrderedContract::Kind::Contraction};
  std::vector<const Meta*> output;
  flatten(*d.output, output);
  if (output.size() != 3) return {};
  std::set<std::size_t> group_ids;
  for (auto* m : output)
    if (m->kind != Meta::Kind::Group || m->index >= d.groups.size() ||
        !group_ids.insert(m->index).second)
      return {};
  for (std::size_t g = 0; g < 4; ++g)
    if (!group_ids.contains(g)) c.contract_group = g;
  c.batch_group = output[0]->index;
  c.left_free_group = output[1]->index;
  c.right_free_group = output[2]->index;
  auto paired = [&](std::size_t g, std::size_t& left, std::size_t& right) {
    const auto& group = d.groups[g];
    if (group.domain || group.bindings.size() != 2) return false;
    std::set<std::size_t> seen;
    for (const auto& b : group.bindings) {
      if (b.operand > 1 || !seen.insert(b.operand).second ||
          b.positions->kind != Meta::Kind::Attribute ||
          b.positions->type != Type::Positions || b.positions->index >= 4)
        return false;
      (b.operand == 0 ? left : right) = b.positions->index;
    }
    return true;
  };
  if (!paired(c.batch_group, c.left_batch, c.right_batch) ||
      !paired(c.contract_group, c.left_contract, c.right_contract) ||
      std::set<std::size_t>{c.left_batch, c.right_batch, c.left_contract,
                            c.right_contract}
              .size() != 4)
    return {};
  auto free = [&](std::size_t g, std::size_t o, std::size_t b, std::size_t k) {
    const auto& group = d.groups[g];
    return !group.domain && group.bindings.size() == 1 &&
           group.bindings[0].operand == o &&
           freePositions(*group.bindings[0].positions, o, b, k);
  };
  if (!free(c.left_free_group, 0, c.left_batch, c.left_contract) ||
      !free(c.right_free_group, 1, c.right_batch, c.right_contract))
    return {};
  return c;
}

}  // namespace tepl::core::axis
