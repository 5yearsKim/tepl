#include "src/rewrite/matcher.h"

#include <functional>

namespace tepl::rewrite {
namespace {
template <class Map, class Key, class T>
bool bind(Map& map, Key key, T value) {
  auto [it, added] = map.emplace(key, value);
  return added || it->second == value;
}
}  // namespace
bool match(const core::Program& p, const core::Rule& r,
           const core::ConcreteGraph& g, std::size_t root, Bindings& b,
           Budget& budget) {
  std::function<bool(const core::Pattern&, std::size_t)> visit =
      [&](const auto& pattern, std::size_t id) {
        budget.tick();
        const auto& n = g.nodes.at(id);
        return std::visit(
            [&](const auto& v) -> bool {
              using T = std::decay_t<decltype(v)>;
              if constexpr (std::is_same_v<T, core::CapturePattern>)
                return bind(b.captures, v.capture.value, id);
              else if constexpr (std::is_same_v<T, core::BindPattern>)
                return bind(b.captures, v.capture.value, id) &&
                       visit(*v.expression, id);
              else if constexpr (std::is_same_v<T, core::GraphLiteral>) {
                if (n.kind != core::ConcreteNode::Kind::kLiteral ||
                    (v.dtype && n.dtype != v.dtype))
                  return false;
                bool decimal = n.text.find('.') != std::string::npos;
                if (decimal != (v.kind == core::GraphLiteral::Kind::kDecimal))
                  return false;
                return decimal
                           ? std::stod(v.spelling) == std::stod(n.text)
                           : parseInteger(v.spelling) == parseInteger(n.text);
              } else {
                if (n.kind != core::ConcreteNode::Kind::kOperation ||
                    n.operation != v.operation ||
                    n.operands.size() != v.operands.size())
                  return false;
                if (v.descriptor) {
                  auto schema = p.operations.at(v.operation.value).attributes;
                  if (!schema) return false;
                  DescriptorValue attrs{schema->value, {}};
                  for (const auto& a : n.attributes)
                    attrs.fields.push_back(attributeValue(a));
                  if (!bind(b.descriptors, v.descriptor->value, attrs))
                    return false;
                }
                for (std::size_t i = 0; i < v.operands.size(); ++i)
                  if (!visit(*v.operands[i], n.operands[i])) return false;
                return true;
              }
            },
            pattern.value);
      };
  if (!visit(*r.lhs, root)) return false;
  for (const auto& c : r.constraints) {
    const auto& n = g.nodes.at(b.captures.at(c.capture.value));
    if (!n.shape) return false;
    std::size_t fixed = 0;
    bool sequence = false;
    for (const auto& e : c.shape)
      if (e.kind == core::ShapeElement::Kind::kSequence)
        sequence = true;
      else
        ++fixed;
    if (sequence ? n.shape->size() < fixed : n.shape->size() != fixed)
      return false;
    std::size_t at = 0;
    for (const auto& e : c.shape) {
      Value value;
      if (e.kind == core::ShapeElement::Kind::kSequence) {
        List xs;
        auto size = n.shape->size() - fixed;
        for (std::size_t i = 0; i < size; ++i)
          xs.emplace_back(Integer(n.shape->at(at++)));
        value = xs;
      } else
        value = Integer(n.shape->at(at++));
      if (e.symbol && !bind(b.dimensions, e.symbol->value, value)) return false;
    }
    if (c.dtype) {
      if (!n.dtype) return false;
      if (const auto* dtype = std::get_if<core::DType>(&*c.dtype)) {
        if (*n.dtype != *dtype) return false;
      } else if (!bind(b.dtypes,
                       std::get<core::DTypeVariableId>(*c.dtype).value,
                       Value(std::string(core::dtypeName(*n.dtype)))))
        return false;
    }
  }
  return true;
}
}  // namespace tepl::rewrite
