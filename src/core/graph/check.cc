#include "src/core/graph/check.h"

#include <charconv>
#include <map>
#include <set>

#include "src/core/resolution/source.h"
#include "src/core/type/literal.h"

namespace tepl::core::detail {
namespace {
class Checker {
 public:
  Checker(AnalysisContext& context, const ast::ConcreteGraph& input)
      : context_(context), input_(input) {
    graph_.id = GraphId{context.output.graphs.size()};
    graph_.name = input.name;
    graph_.source_name = input.source_name;
    graph_.origin = at(input.span);
  }
  void run() {
    const auto errors = context_.diagnostics.size();
    for (const auto& input : input_.inputs) {
      ConcreteNode node{ConcreteNode::Kind::kInput, at(input.span), input.name};
      node.dtype = dtype(input.dtype, input.span);
      if (input.shape) {
        node.shape.emplace();
        for (const auto& spelling : *input.shape) {
          std::uint64_t number = 0;
          const auto parsed = std::from_chars(
              spelling.data(), spelling.data() + spelling.size(), number);
          if (parsed.ec != std::errc{} ||
              parsed.ptr != spelling.data() + spelling.size())
            error(input.span, "graph dimension is outside the index range");
          node.shape->push_back(number);
        }
      }
      auto id = graph_.nodes.size();
      graph_.nodes.push_back(std::move(node));
      bind(input.name, id, input.span);
    }
    for (const auto& binding : input_.bindings) {
      auto id = expression(binding.expression);
      if (id) bind(binding.name, *id, binding.span);
    }
    auto root = expression(input_.output);
    if (root && errors == context_.diagnostics.size()) {
      graph_.root = *root;
      context_.output.graphs.push_back(std::move(graph_));
    }
  }

 private:
  SourceOrigin at(SourceSpan span) const {
    return origin(input_.source_name, span);
  }
  void error(SourceSpan span, const std::string& message) {
    context_.report(at(span), message);
  }
  std::optional<DType> dtype(
      const std::optional<ast::DTypeAnnotation>& annotation, SourceSpan span) {
    if (!annotation) return {};
    auto value = resolveDType(annotation->name);
    if (!value) error(span, "unknown dtype '" + annotation->name + "'");
    return value;
  }
  void bind(const std::string& name, std::size_t id, SourceSpan span) {
    if (!bindings_.emplace(name, id).second)
      error(span, "duplicate graph value '" + name + "'");
    else
      graph_.bindings.emplace_back(name, id);
  }
  AttributeValue attribute(const ast::ConcreteValue& input,
                           const AttributeField& field, std::size_t depth,
                           bool optional) {
    using A = ast::ConcreteValue::Kind;
    using K = AttributeValue::Kind;
    if (optional && input.kind == A::kName && input.text == "none")
      return {K::kNone, {}, {}};
    if (depth) {
      AttributeValue value{K::kList, {}, {}};
      if (input.kind != A::kList) {
        error(input.span, "attribute '" + field.name + "' expects a list");
        return value;
      }
      for (const auto& child : input.elements)
        value.elements.push_back(attribute(child, field, depth - 1, false));
      return value;
    }
    const auto bad = [&] {
      error(input.span, "invalid value for attribute '" + field.name +
                            "' of type '" + field.type + "'");
    };
    if (field.type == "index" || field.type == "i64") {
      auto type = field.type == "index" ? TypeKind::kIndex : TypeKind::kI64;
      if (input.kind != A::kInteger || !validHostLiteral(input.text, type))
        bad();
      // Canonical integer text avoids octal C++ constants and target parser
      // limits.
      auto text = input.text;
      bool negative = text.starts_with('-');
      if (text.starts_with('+') || negative) text.erase(0, 1);
      auto first = text.find_first_not_of('0');
      text = first == std::string::npos ? "0" : text.substr(first);
      if (negative && text != "0") text = "-" + text;
      return {K::kInteger, text, {}};
    }
    if (field.type == "bool") {
      if (input.kind != A::kBool) bad();
      return {K::kBool, input.text, {}};
    }
    if (field.type == "string") {
      if (input.kind != A::kString) bad();
      return {K::kString, input.text, {}};
    }
    if (field.type == "dtype") {
      if (input.kind != A::kName || !resolveDType(input.text)) bad();
      return {K::kEnum, input.text, {}};
    }
    if (field.type == "precision") {
      if (input.kind != A::kName ||
          (input.text != "default" && input.text != "high" &&
           input.text != "highest"))
        bad();
      return {K::kEnum, input.text, {}};
    }
    error(input.span, "graph attribute type '" + field.type +
                          "' has no concrete value syntax yet");
    return {K::kNone, {}, {}};
  }
  std::optional<std::size_t> expression(const ast::ConcreteExpr& expr) {
    using K = ast::ConcreteExpr::Kind;
    if (expr.kind == K::kReference) {
      auto found = bindings_.find(expr.name);
      if (found == bindings_.end()) {
        error(expr.span,
              "unknown or forward graph reference '" + expr.name + "'");
        return {};
      }
      return found->second;
    }
    ConcreteNode node{expr.kind == K::kLiteral ? ConcreteNode::Kind::kLiteral
                                               : ConcreteNode::Kind::kOperation,
                      at(expr.span), expr.name};
    if (expr.kind == K::kLiteral) {
      node.dtype = dtype(expr.dtype, expr.span);
      if (node.dtype && !validGraphLiteral(expr.name, *node.dtype))
        error(expr.span,
              "literal '" + expr.name + "' is invalid for its dtype");
    } else {
      node.operation =
          context_.operation(expr.name, input_.source_name, at(expr.span));
      if (!node.operation) return {};
      const auto& op = context_.output.operations.at(node.operation->value);
      if (context_.types.known(op.result)->kind != TypeKind::kTensor)
        error(expr.span, "graph operations must return tensors");
      bool variadic = !op.operands.empty() && op.operands.back().variadic;
      auto minimum = op.operands.size() - (variadic ? 1 : 0);
      if (expr.operands.size() < minimum ||
          (!variadic && expr.operands.size() != minimum))
        error(expr.span,
              "invalid operand count for graph operation '" + expr.name + "'");
      for (const auto& operand : op.operands)
        if (context_.types.known(operand.type)->kind != TypeKind::kTensor)
          error(expr.span, "graph operation operands must be tensors");
      std::map<std::string, const ast::ConcreteValue*> fields;
      for (const auto& field : expr.attributes)
        if (!fields.emplace(field.name, &field.value).second)
          error(field.span, "duplicate graph attribute '" + field.name + "'");
      if (!op.attributes && !fields.empty())
        error(expr.span, "operation '" + expr.name + "' has no attributes");
      if (op.attributes) {
        const auto& schema =
            context_.output.attribute_schemas.at(op.attributes->value);
        for (const auto& field : schema.fields) {
          auto found = fields.find(field.name);
          if (found != fields.end()) {
            node.attributes.push_back(attribute(
                *found->second, field, field.list_depth, field.optional));
            fields.erase(found);
          } else if (field.empty_default)
            node.attributes.push_back({AttributeValue::Kind::kList, {}, {}});
          else if (field.optional)
            node.attributes.push_back({AttributeValue::Kind::kNone, {}, {}});
          else
            error(expr.span, "missing graph attribute '" + field.name + "'");
        }
        for (const auto& [name, value] : fields)
          error(value->span, "unknown graph attribute '" + name + "'");
      }
      for (const auto& operand : expr.operands) {
        auto id = expression(operand);
        if (!id) return {};
        node.operands.push_back(*id);
      }
    }
    auto id = graph_.nodes.size();
    graph_.nodes.push_back(std::move(node));
    return id;
  }
  AnalysisContext& context_;
  const ast::ConcreteGraph& input_;
  ConcreteGraph graph_;
  std::map<std::string, std::size_t> bindings_;
};
}  // namespace
void checkGraphs(AnalysisContext& context) {
  std::set<std::pair<std::string, std::string>> names;
  for (const auto& graph : context.input.graphs) {
    if (!names.emplace(sourceKey(graph.source_name), graph.name).second)
      context.report(origin(graph.source_name, graph.span),
                     "duplicate graph '" + graph.name + "'");
    else
      Checker(context, graph).run();
  }
}
}  // namespace tepl::core::detail
