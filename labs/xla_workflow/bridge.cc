#include "labs/xla_workflow/bridge.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <functional>
#include <stdexcept>

#include "src/executable/program.h"
#include "src/rewrite/builder.h"
#include "src/rewrite/matcher.h"
using namespace tepl;
namespace {
struct Context {
  executable::ExecutableProgram program;
  core::ConcreteGraph graph;
  std::vector<WorkflowNode> views;
  std::vector<std::array<std::vector<int64_t>, 6>> lists;
  explicit Context(executable::ExecutableProgram p) : program(std::move(p)) {}
};
void errorText(char* out, size_t n, const char* what) {
  if (out && n) std::snprintf(out, n, "%s", what);
}
std::string read(const char* path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("missing program");
  return {std::istreambuf_iterator<char>(f), {}};
}
core::AttributeValue attribute(WorkflowList l) {
  core::AttributeValue a{core::AttributeValue::Kind::kList, {}, {}};
  for (size_t i = 0; i < l.size; ++i)
    a.elements.push_back(
        {core::AttributeValue::Kind::kInteger, std::to_string(l.data[i]), {}});
  return a;
}
const char* names[] = {"lhs_contracting_dimensions",
                       "rhs_contracting_dimensions", "lhs_batching_dimensions",
                       "rhs_batching_dimensions"};
}  // namespace
extern "C" void* tepl_workflow_load(const char* p, char* e, size_t n) {
  try {
    return new Context(executable::load(read(p)));
  } catch (const std::exception& x) {
    errorText(e, n, x.what());
    return nullptr;
  }
}
extern "C" int tepl_workflow_match(void* opaque, const WorkflowNode* nodes,
                                   size_t count, size_t root, uint64_t excluded,
                                   WorkflowResult* result, char* error,
                                   size_t error_size) {
  try {
    auto& c = *static_cast<Context*>(opaque);
    const auto& p = c.program.ir();
    c.graph = {};
    c.graph.name = "hlo_view";
    c.graph.root = root;
    if (root >= count) throw std::runtime_error("bad root");
    for (size_t i = 0; i < count; ++i) {
      const auto& in = nodes[i];
      core::ConcreteNode node{};
      node.kind = core::ConcreteNode::Kind::kInput;
      node.text = "opaque";
      node.dtype = core::DType::kF32;
      node.shape = std::vector<uint64_t>();
      for (size_t k = 0; k < in.shape.size; ++k)
        node.shape->push_back(in.shape.data[k]);
      if (in.op && in.f32) {
        const char* name = in.op == 1 ? "transpose" : "dot_general";
        const executable::Operation* operation = nullptr;
        for (const auto& op : p.operations)
          if (op.name == name) operation = &op;
        if (!operation) throw std::runtime_error("operation not registered");
        node.kind = core::ConcreteNode::Kind::kOperation;
        node.operation = operation->id;
        for (size_t k = 0; k < in.arity; ++k) {
          if (in.operands[k] >= i)
            throw std::runtime_error("non-topological graph");
          node.operands.push_back(in.operands[k]);
        }
        if (in.op == 1)
          node.attributes.push_back(attribute(in.permutation));
        else
          for (const auto& field :
               p.attribute_schemas.at(operation->attributes->value).fields) {
            size_t k = 0;
            while (k < 4 && field.name != names[k]) ++k;
            if (k == 4) throw std::runtime_error("unknown field");
            node.attributes.push_back(attribute(in.fields[k]));
          }
      }
      c.graph.nodes.push_back(std::move(node));
    }
    if (!nodes[root].f32) return 0;
    for (const auto& rule : p.rules) {
      if (rule.id.value >= 64) throw std::runtime_error("too many rules");
      if (excluded & (uint64_t(1) << rule.id.value)) continue;
      rewrite::Budget budget;
      rewrite::Bindings bindings;
      if (!rewrite::match(p, rule, c.graph, root, bindings, budget)) continue;
      bool supported = true;
      for (const auto& entry : bindings.captures)
        if (!nodes[entry.second].f32) supported = false;
      if (!supported) continue;
      // Match before contract evaluation: unrelated HLO instructions are
      // opaque.
      for (size_t i = 0; i < count; ++i) rewrite::infer(p, c.graph, i, budget);
      rewrite::Evaluator eval(c.program, c.graph, rule.id.value, bindings,
                              budget);
      if (!eval.evaluateWhere()) continue;
      eval.constructAttributes();
      size_t newroot =
          rewrite::build(p, *rule.rhs, c.graph, bindings, budget, 10000);
      c.views.assign(c.graph.nodes.size(), {});
      c.lists.clear();
      c.lists.resize(c.views.size());
      for (size_t i = 0; i < c.views.size(); ++i) {
        const auto& node = c.graph.nodes[i];
        auto& view = c.views[i];
        auto& ls = c.lists[i];
        view.original_id = i < count ? nodes[i].original_id : -1;
        view.f32 = 1;
        for (auto dim : *node.shape) ls[0].push_back(dim);
        view.shape = {ls[0].data(), ls[0].size()};
        if (node.kind == core::ConcreteNode::Kind::kOperation) {
          const auto& op = p.operations.at(node.operation->value);
          view.op = op.name == "transpose" ? 1 : 2;
          view.arity = node.operands.size();
          if (view.arity > 2) throw std::runtime_error("adapter arity");
          for (size_t k = 0; k < view.arity; ++k)
            view.operands[k] = node.operands[k];
          const auto& schema = p.attribute_schemas.at(op.attributes->value);
          for (size_t j = 0; j < schema.fields.size(); ++j) {
            size_t k = 0;
            if (view.op == 2) {
              while (k < 4 && schema.fields[j].name != names[k]) ++k;
            }
            auto& xs = ls[view.op == 1 ? 1 : k + 2];
            for (const auto& v : node.attributes[j].elements)
              xs.push_back(std::stoll(v.text));
            WorkflowList list{xs.data(), xs.size()};
            if (view.op == 1)
              view.permutation = list;
            else
              view.fields[k] = list;
          }
        }
      }
      *result = {c.views.data(), c.views.size(), newroot, rule.id.value};
      return 1;
    }
    return 0;
  } catch (const std::exception& x) {
    errorText(error, error_size, x.what());
    return -1;
  }
}
extern "C" void tepl_workflow_destroy(void* c) {
  delete static_cast<Context*>(c);
}

extern "C" uint64_t tepl_workflow_roots(void* opaque) {
  auto& p = static_cast<Context*>(opaque)->program.ir();
  uint64_t mask = 0;
  std::function<void(const core::Pattern&)> visit = [&](const auto& pattern) {
    std::visit(
        [&](const auto& v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, core::BindPattern>)
            visit(*v.expression);
          else if constexpr (std::is_same_v<T, core::MatchOperation>) {
            const auto& name = p.operations.at(v.operation.value).name;
            if (name == "transpose")
              mask |= 2;
            else if (name == "dot_general")
              mask |= 4;
          }
        },
        pattern.value);
  };
  for (const auto& r : p.rules) visit(*r.lhs);
  return mask;
}
extern "C" size_t tepl_workflow_depth(void* opaque) {
  auto& p = static_cast<Context*>(opaque)->program.ir();
  std::function<size_t(const core::Pattern&)> depth = [&](const auto& pattern) {
    return std::visit(
        [&](const auto& v) -> size_t {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, core::BindPattern>)
            return depth(*v.expression);
          else if constexpr (std::is_same_v<T, core::MatchOperation>) {
            size_t d = 0;
            for (const auto& a : v.operands) d = std::max(d, depth(*a));
            return d + 1;
          } else
            return 0;
        },
        pattern.value);
  };
  size_t d = 0;
  for (const auto& r : p.rules) d = std::max(d, depth(*r.lhs));
  return d;
}
