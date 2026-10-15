#include "src/core/semantics/completion.h"

#include <stdexcept>
namespace tepl::core::semantics {
completed::SymbolPtr Completion::symbol(const symbolic::SymbolPtr& s) {
  if (!s) throw std::invalid_argument("null materialized symbol");
  if (auto i = symbols_.find(s.get()); i != symbols_.end())
    return i->second.second;
  completed::Symbol out;
  using K = symbolic::Symbol::Kind;
  using C = completed::Symbol::Kind;
  switch (s->kind) {
#define KIND(name)      \
  case K::name:         \
    out.kind = C::name; \
    break
    KIND(Integer);
    KIND(Boolean);
    KIND(Shape);
    KIND(DType);
    KIND(Field);
    KIND(AxisGroup);
    KIND(List);
    KIND(Call);
#undef KIND
    case K::Hole:
      throw std::invalid_argument("unresolved symbolic hole");
  }
  out.name = s->name;
  out.owner = s->owner;
  out.index = s->index;
  out.sort = static_cast<completed::Symbol::Sort>(s->sort);
  for (const auto& a : s->args) out.args.push_back(symbol(a));
  auto result = std::make_shared<const completed::Symbol>(std::move(out));
  symbols_.emplace(s.get(), std::pair{s, result});
  return result;
}
completed::TypedExprPtr Completion::expression(const TypedExprPtr& e) {
  if (!e) throw std::invalid_argument("null materialized expression");
  if (auto i = expressions_.find(e.get()); i != expressions_.end())
    return i->second;
  completed::TypedExpr out{e->origin, e->type};
  std::visit(
      [&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        auto list = [&](const auto& input) {
          std::vector<completed::TypedExprPtr> result;
          for (const auto& x : input) result.push_back(expression(x));
          return result;
        };
        if constexpr (std::is_same_v<T, OutputAxisGroupRef>) {
          if (!groups_)
            throw std::invalid_argument("unresolved output-axis reference");
          out.value = completed::SymbolicExpr{symbol(groups_(v))};
        } else if constexpr (std::is_same_v<T, BuiltinCall>)
          out.value = completed::BuiltinCall{v.builtin, list(v.arguments)};
        else if constexpr (std::is_same_v<T, UnaryExpr>)
          out.value = completed::UnaryExpr{v.op, expression(v.operand)};
        else if constexpr (std::is_same_v<T, BinaryExpr>)
          out.value =
              completed::BinaryExpr{v.op, expression(v.lhs), expression(v.rhs)};
        else if constexpr (std::is_same_v<T, LocalLet>)
          out.value = completed::LocalLet{v.local, expression(v.value)};
        else if constexpr (std::is_same_v<T, ListExpr>)
          out.value = completed::ListExpr{list(v.elements)};
        else if constexpr (std::is_same_v<T, IndexExpr>)
          out.value =
              completed::IndexExpr{expression(v.value), expression(v.index)};
        else if constexpr (std::is_same_v<T, ConditionalExpr>)
          out.value = completed::ConditionalExpr{expression(v.condition),
                                                 expression(v.then_value),
                                                 expression(v.else_value)};
        else if constexpr (std::is_same_v<T, ComprehensionExpr>)
          out.value = completed::ComprehensionExpr{
              expression(v.element), v.variable, expression(v.iterable)};
        else if constexpr (std::is_same_v<T, OptionalValue>)
          out.value = completed::OptionalValue{expression(v.value)};
        else if constexpr (std::is_same_v<T, DescriptorConstruct>)
          out.value = completed::DescriptorConstruct{v.schema, list(v.fields)};
        else if constexpr (std::is_same_v<T, SymbolicExpr>)
          out.value = completed::SymbolicExpr{symbol(v.value)};
        else
          out.value = v;
      },
      e->value);
  auto result = std::make_shared<const completed::TypedExpr>(std::move(out));
  expressions_.emplace(e.get(), result);
  return result;
}
completed::RuleData Completion::rule(RuleData r) {
  completed::RuleData out;
  out.source_name = std::move(r.source_name);
  out.id = r.id;
  out.name = std::move(r.name);
  out.origin = std::move(r.origin);
  out.captures = std::move(r.captures);
  out.dimensions = std::move(r.dimensions);
  out.lhs = std::move(r.lhs);
  out.rhs = std::move(r.rhs);
  out.constraints = std::move(r.constraints);
  out.locals = std::move(r.locals);
  out.dtypes = std::move(r.dtypes);
  for (auto& d : r.descriptors) {
    using K = Descriptor::Kind;
    using C = completed::Descriptor::Kind;
    if (d.kind == K::kInferred)
      throw std::invalid_argument("unresolved descriptor inference");
    out.descriptors.push_back(
        {d.id, std::move(d.name),
         d.kind == K::kCaptured ? C::kCaptured : C::kDerived, d.type,
         std::move(d.origin)});
  }
  for (const auto& c : r.conditions) out.conditions.push_back(expression(c));
  for (auto& d : r.derivations)
    out.derivations.push_back(
        {d.target, expression(d.value), std::move(d.origin)});
  return out;
}
}  // namespace tepl::core::semantics
