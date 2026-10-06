#pragma once
#include <eggc/egraph.hpp>
#include <map>

#include "../analysis/analysis.h"

namespace @TEPL_NAMESPACE@::graphs {
struct InputSpec {
  ::std::size_t node;
  ::std::string name;
  ::std::optional<::std::vector<::std::uint64_t>> shape;
  ::std::optional<DType> dtype;
};
struct BuiltGraph {
  ::eggc::Id root;
  ::std::map<::std::string, ::eggc::Id, ::std::less<>> bindings;
  ::std::optional<::eggc::Id> get(::std::string_view name) const {
    auto found = bindings.find(name);
    return found == bindings.end() ? ::std::nullopt
                                   : ::std::optional(found->second);
  }
};
class GraphDefinition {
 public:
  GraphDefinition(
      ::std::vector<OpNode> nodes, ::std::vector<InputSpec> inputs,
      ::std::map<::std::string, ::std::size_t, ::std::less<>> bindings,
      ::std::size_t root)
      : nodes_(::std::move(nodes)),
        inputs_(::std::move(inputs)),
        bindings_(::std::move(bindings)),
        root_(root) {
    validate();
  }
  const ::std::vector<OpNode>& nodes() const { return nodes_; }
  const ::std::vector<InputSpec>& inputs() const { return inputs_; }
  ::std::size_t root() const { return root_; }
  ::std::optional<::std::size_t> get(::std::string_view name) const {
    auto found = bindings_.find(name);
    return found == bindings_.end() ? ::std::nullopt
                                    : ::std::optional(found->second);
  }
  GraphDefinition with_input_info(::std::string_view name,
                                  TensorInfo info) const {
    auto result = *this;
    for (auto& input : result.inputs_) {
      if (input.name != name) continue;
      if ((input.shape && *input.shape != info.shape) ||
          (input.dtype && *input.dtype != info.dtype))
        throw NodeError("conflicting graph input metadata");
      input.shape = info.shape;
      input.dtype = info.dtype;
      result.validate();
      return result;
    }
    throw NodeError("unknown graph input");
  }
  analysis::TensorBindingTable input_bindings() const {
    analysis::TensorBindingTable bindings;
    for (const auto& input : inputs_)
      if (input.shape && input.dtype)
        bindings.register_symbol(input.name, {*input.shape, *input.dtype});
    return bindings;
  }
  template <class A>
  BuiltGraph insert_into(::eggc::EGraph<OpNode, A>& graph) const {
    for (const auto& input : inputs_)
      if (input.shape || input.dtype)
        throw NodeError(
            "graph input metadata requires into_egraph or into_egraph_with");
    return insert_nodes(graph);
  }
  template <class Factory>
  auto into_egraph_with(Factory factory) const {
    using A = decltype(factory(input_bindings()));
    ::eggc::EGraph<OpNode, A> graph(factory(input_bindings()));
    auto built = insert_nodes(graph);
    return ::std::make_pair(::std::move(graph), ::std::move(built));
  }
  auto into_egraph() const {
    return into_egraph_with([](analysis::TensorBindingTable inputs) {
      return analysis::TensorAnalysis(::std::move(inputs));
    });
  }

 private:
  void validate() const {
    if (root_ >= nodes_.size()) throw NodeError("invalid graph root");
    for (const auto& [name, id] : bindings_)
      if (id >= nodes_.size()) throw NodeError("invalid graph binding");
    using Shapes = ::std::vector<::std::uint64_t>;
    ::std::vector<::std::optional<Shapes>> shapes(nodes_.size());
    ::std::vector<::std::optional<DType>> dtypes(nodes_.size());
    ::std::map<::std::size_t, const InputSpec*> input_nodes;
    ::std::map<::std::string, ::std::size_t> input_names;
    for (const auto& input : inputs_) {
      if (input.node >= nodes_.size() ||
          !(nodes_[input.node] == OpNode::input(input.name)) ||
          !input_nodes.emplace(input.node, &input).second ||
          !input_names.emplace(input.name, input.node).second)
        throw NodeError("invalid or duplicate graph input");
      shapes[input.node] = input.shape;
      dtypes[input.node] = input.dtype;
    }
    for (::std::size_t index = 0; index < nodes_.size(); ++index) {
      const auto& node = nodes_[index];
      for (auto child : node.children())
        if (child >= index)
          throw NodeError("graph operands must refer to earlier nodes");
      OpNode::from_parts(node.op(), node.children(), node.attrs());
      if (node.op() == Op::Input) {
        if (!input_nodes.contains(index))
          throw NodeError("undeclared graph input");
        continue;
      }
      if (auto literal = ::std::get_if<LiteralAttrs>(&node.attrs().value)) {
        shapes[index] = Shapes{};
        dtypes[index] = literal->dtype;
        continue;
      }
      bool known_shape = true, known_dtype = true;
      ::std::vector<Shapes> operand_shapes;
      ::std::vector<DType> operand_dtypes;
      for (auto child : node.children()) {
        if (shapes[child])
          operand_shapes.push_back(*shapes[child]);
        else
          known_shape = false;
        if (dtypes[child])
          operand_dtypes.push_back(*dtypes[child]);
        else
          known_dtype = false;
      }
      if (known_shape) {
        auto result =
            analysis::infer_shape(node.op(), operand_shapes, node.attrs());
        if (result.kind == analysis::Inference<Shapes>::Kind::Invalid)
          throw NodeError("graph node " + ::std::to_string(index) + ": " +
                          result.error);
        shapes[index] = result.value;
      }
      if (known_dtype) {
        auto result =
            analysis::infer_dtype(node.op(), operand_dtypes, node.attrs());
        if (result.kind == analysis::Inference<DType>::Kind::Invalid)
          throw NodeError("graph node " + ::std::to_string(index) + ": " +
                          result.error);
        dtypes[index] = result.value;
      }
    }
  }
  template <class A>
  BuiltGraph insert_nodes(::eggc::EGraph<OpNode, A>& graph) const {
    validate();
    ::std::vector<::eggc::Id> ids;
    for (const auto& node : nodes_) {
      ::std::vector<::eggc::Id> children;
      for (auto child : node.children()) children.push_back(ids.at(child));
      ids.push_back(graph.add(
          OpNode::from_parts(node.op(), ::std::move(children), node.attrs())));
    }
    graph.rebuild();
    BuiltGraph built{graph.find(ids[root_]), {}};
    for (const auto& [name, id] : bindings_)
      built.bindings.emplace(name, graph.find(ids[id]));
    return built;
  }
  ::std::vector<OpNode> nodes_;
  ::std::vector<InputSpec> inputs_;
  ::std::map<::std::string, ::std::size_t, ::std::less<>> bindings_;
  ::std::size_t root_;
};
}  // namespace @TEPL_NAMESPACE@::graphs

namespace @TEPL_NAMESPACE@ {
using graphs::BuiltGraph;
using graphs::GraphDefinition;
using graphs::InputSpec;
}  // namespace @TEPL_NAMESPACE@
