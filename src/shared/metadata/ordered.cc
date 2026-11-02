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
bool sameMetadata(const Meta& a, const Meta& b) {
  if(a.kind!=b.kind || a.type!=b.type || a.name!=b.name || a.index!=b.index || a.args.size()!=b.args.size()) return false;
  for(std::size_t i=0;i<a.args.size();++i)if(!sameMetadata(*a.args[i],*b.args[i]))return false;
  return true;
}
bool unitSizes(const Meta& m) {
  return named(m,"gather",2) && named(*m.args[0],"shape",1) &&
      operand(*m.args[0]->args[0],0) && field(*m.args[1],0);
}
std::optional<OrderedContract> unitRemoval(const ShapeContract& d) {
  if(d.operands.size()!=1 || d.fields.size()!=1 || d.groups.size()!=2 ||
      !d.extent_fields.empty() || d.implicit_operands!=std::set<std::size_t>{0} ||
      d.assertions.size()!=3 || !d.output || d.output->kind!=Meta::Kind::Group || d.output->index>=2)return {};
  OrderedContract c{OrderedContract::Kind::UnitRemoval};
  c.left_free_group=d.output->index;c.contract_group=1-d.output->index;
  const auto& u=d.groups[c.contract_group];const auto& f=d.groups[c.left_free_group];
  if(u.domain || f.domain || u.bindings.size()!=1 || f.bindings.size()!=1 ||
      u.bindings[0].operand!=0 || f.bindings[0].operand!=0 || !field(*u.bindings[0].positions,0))return {};
  const auto& retained=*f.bindings[0].positions;
  if(!named(retained,"ordered_complement",2) || !all(*retained.args[0],0) || !field(*retained.args[1],0))return {};
  bool valid=false,positive=false,one=false;
  for(const auto& a:d.assertions) {
    if(named(*a,"is_valid_axis_list",2) && field(*a->args[0],0) && rank(*a->args[1],0))valid=true;
    else if(named(*a,"is_positive_extents",1) && unitSizes(*a->args[0]))positive=true;
    else if(named(*a,"extent_le",2) && unitSizes(*a->args[0])) {
      const auto& rhs=*a->args[1];
      if(!named(rhs,"offset_extents",2) || rhs.args[1]->kind!=Meta::Kind::Integer || rhs.args[1]->name!="1")return {};
      const auto& zero=*rhs.args[0];
      if(!named(zero,"extent_sub",2) || !sameMetadata(*a->args[0],*zero.args[0]) || !sameMetadata(*zero.args[0],*zero.args[1]))return {};
      one=true;
    }else return {};
  }
  return valid && positive && one ? std::optional{c}:std::nullopt;
}
}  // namespace

std::optional<OrderedContract> recognizeOrdered(const ShapeContract& d) {
  if (!d.output || !d.extent_fields.empty()) return {};
  if (d.operands.size() == 2 && d.fields.empty() && d.groups.size() == 1 &&
      d.assertions.empty() &&
      d.implicit_operands == std::set<std::size_t>{0, 1} &&
      d.output->kind == Meta::Kind::Group && d.output->index == 0) {
    const auto& group = d.groups[0];
    if (group.domain || group.bindings.size() != 2) return {};
    std::set<std::size_t> operands;
    for (const auto& binding : group.bindings)
      if (binding.operand > 1 || !operands.insert(binding.operand).second ||
          !all(*binding.positions, binding.operand))
        return {};
    return OrderedContract{OrderedContract::Kind::Addition};
  }
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
  // Whole-axis unary reduction. No extra assertions or domain/extent fields:
  // the selected axes are unrestricted, and the complement preserves order.
  if (d.operands.size() == 1 && d.fields.size() == 1 && d.groups.size() == 2 &&
      d.assertions.empty() && d.implicit_operands == std::set<std::size_t>{0} &&
      d.output->kind == Meta::Kind::Group && d.output->index < 2) {
    OrderedContract c{OrderedContract::Kind::Summation};
    c.left_free_group = d.output->index;
    c.contract_group = 1 - c.left_free_group;
    const auto& reduced = d.groups[c.contract_group];
    const auto& retained = d.groups[c.left_free_group];
    if (reduced.domain || retained.domain || reduced.bindings.size() != 1 ||
        retained.bindings.size() != 1 || reduced.bindings[0].operand != 0 ||
        retained.bindings[0].operand != 0 ||
        !field(*reduced.bindings[0].positions, 0))
      return {};
    const auto& positions = *retained.bindings[0].positions;
    if (!named(positions, "ordered_complement", 2) ||
        !all(*positions.args[0], 0) || !field(*positions.args[1], 0))
      return {};
    return c;
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

std::optional<OrderedContract> recognizeCoreShape(const ShapeContract& d) {
  if (auto c = recognizeOrdered(d)) return c;
  if (auto c = unitRemoval(d)) return c;
  if (!d.output || d.operands.size() != 1) return {};
  if (d.fields.empty() && d.groups.size() == 1 && d.extent_fields.empty() &&
      d.assertions.empty() && d.implicit_operands == std::set<std::size_t>{0} &&
      !d.groups[0].domain && d.groups[0].bindings.size() == 1 &&
      d.groups[0].bindings[0].operand == 0 && all(*d.groups[0].bindings[0].positions, 0) &&
      d.output->kind == Meta::Kind::Group && d.output->index == 0)
    return OrderedContract{OrderedContract::Kind::Negation};
  auto group = [&](std::size_t i) {
    return i < d.groups.size() && !d.groups[i].domain &&
        d.groups[i].bindings.size() == 1 &&
        d.groups[i].bindings[0].operand == 0 &&
        all(*d.groups[i].bindings[0].positions, 0);
  };
  if (d.implicit_operands.empty() && !d.assertions.empty() && d.fields.size() == 1 && d.extent_fields.empty() &&
      d.groups.size() == 1 &&
      group(0) && d.output->kind == Meta::Kind::Group && d.output->index == 0) {
    for (const auto& a : d.assertions)
      if (!named(*a, "is_valid_axis_list", 2) ||
          !field(*a->args[0], 0) || !rank(*a->args[1], 0)) return {};
    return OrderedContract{OrderedContract::Kind::Reverse};
  }
  if (d.implicit_operands == std::set<std::size_t>{0} && d.fields.size() == 2 && d.groups.size() == 2 &&
      d.extent_fields == std::set<std::size_t>{1} &&
      d.assertions.empty() && group(0) &&
      d.groups[1].bindings.empty() && d.groups[1].domain &&
      d.groups[1].domain->kind == Meta::Kind::Attribute &&
      d.groups[1].domain->index == 1 &&
      named(*d.output, "place", 3) &&
      d.output->args[0]->kind == Meta::Kind::Group &&
      d.output->args[0]->index == 0 && field(*d.output->args[1], 0) &&
      d.output->args[2]->kind == Meta::Kind::Group &&
      d.output->args[2]->index == 1)
    return OrderedContract{OrderedContract::Kind::Broadcast};
  return {};
}

}  // namespace tepl::core::axis
