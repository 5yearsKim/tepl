#include "src/core/syntax/snapshot.h"
namespace tepl::syntax {
namespace {
template <class T>
std::shared_ptr<const T> clone(const std::shared_ptr<const T>&);
template <class T>
void children(T& v) {
  if constexpr (requires { v.operands; })
    for (auto& x : v.operands) x = clone(x);
  if constexpr (requires { v.arguments; })
    for (auto& x : v.arguments) x = clone(x);
  if constexpr (requires { v.elements; })
    for (auto& x : v.elements) x = clone(x);
  if constexpr (requires { v.fields; })
    for (auto& f : v.fields) f.value = clone(f.value);
  if constexpr (requires { v.operand; }) v.operand = clone(v.operand);
  if constexpr (requires { v.lhs; }) {
    v.lhs = clone(v.lhs);
    v.rhs = clone(v.rhs);
  }
  if constexpr (requires { v.expression; }) v.expression = clone(v.expression);
  if constexpr (requires { v.tuple; }) v.tuple = clone(v.tuple);
  if constexpr (requires { clone(v.value); }) v.value = clone(v.value);
  if constexpr (requires {
                  v.index;
                  clone(v.index);
                })
    v.index = clone(v.index);
  if constexpr (requires { v.condition; }) v.condition = clone(v.condition);
  if constexpr (requires { v.then_value; }) {
    v.then_value = clone(v.then_value);
    v.else_value = clone(v.else_value);
  }
  if constexpr (requires { v.element; }) v.element = clone(v.element);
  if constexpr (requires { v.iterable; }) v.iterable = clone(v.iterable);
}
template <class T>
std::shared_ptr<const T> clone(const std::shared_ptr<const T>& p) {
  if (!p) return {};
  T out = *p;
  if constexpr (std::is_same_v<T, SemanticExpr>)
    for (auto& a : out.arguments) a = clone(a);
  else
    std::visit([](auto& v) { children(v); }, out.value);
  return std::make_shared<const T>(std::move(out));
}
void metadata(MetadataDefinition& d) {
  for (auto& s : d.statements)
    std::visit([](auto& v) { children(v); }, s.value);
  d.result.value = clone(d.result.value);
}
}  // namespace
Rule snapshot(const Rule& r) {
  auto out = r;
  out.lhs = clone(r.lhs);
  out.rhs = clone(r.rhs);
  for (auto& c : out.conditions) c = clone(c);
  for (auto& d : out.derivations) d.value = clone(d.value);
  return out;
}
ParsedModule snapshot(const ParsedModule& m) {
  auto out = m;
  for (auto& r : out.rules) r = snapshot(r);
  for (auto& d : out.dialects)
    for (auto& op : d.operations) {
      if (op.shape_definition) metadata(*op.shape_definition);
      if (op.dtype_definition) metadata(*op.dtype_definition);
      if (op.semantics) op.semantics->result = clone(op.semantics->result);
      if (op.axes) {
        for (auto& s : op.axes->statements)
          std::visit(
              [](auto& v) {
                if constexpr (std::is_same_v<std::decay_t<decltype(v)>,
                                             AxisBind>) {
                  v.domain = clone(v.domain);
                  for (auto& b : v.bindings) b.positions = clone(b.positions);
                } else
                  std::visit([](auto& x) { children(x); }, v.value);
              },
              s);
        op.axes->output = clone(op.axes->output);
      }
    }
  return out;
}
}  // namespace tepl::syntax
