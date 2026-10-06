#pragma once

#include <algorithm>
#include <eggc/egraph.hpp>
#include <eggc/expr.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

// Each binary aliases its generated namespace as tepl before including this
// file.
namespace utils {
inline void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

inline std::string tensor_type(const tepl::TensorInfo& info) {
  std::string result(tepl::dtype_name(info.dtype));
  result += '[';
  for (std::size_t i = 0; i < info.shape.size(); ++i) {
    if (i) result += ", ";
    result += std::to_string(info.shape[i]);
  }
  return result + ']';
}

inline std::string metadata(const std::monostate&) { return {}; }
inline std::string metadata(const tepl::TensorAnalysisData& data) {
  if (auto info = data.info()) return ": " + tensor_type(*info);
  return data.is_invalid() ? ": invalid metadata" : ": unknown metadata";
}

inline std::string node_name(const tepl::OpNode& node) {
  if (auto input = std::get_if<tepl::InputAttrs>(&node.attrs().value))
    return input->name;
  if (auto literal = std::get_if<tepl::LiteralAttrs>(&node.attrs().value))
    return literal->value +
           (literal->dtype
                ? ":" + std::string(tepl::dtype_name(*literal->dtype))
                : "");
  std::string name(node.op().name());
  namespace d = tepl::dialects::my_dialect;
  if (auto attrs = std::get_if<d::OpAttrs>(&node.attrs().value))
    if (auto dot = std::get_if<d::DotAttrs>(&attrs->value))
      name += "[axis=" + std::to_string(dot->axis) + ']';
  return name;
}

template <class FormatChild>
std::string format_node(const tepl::OpNode& node, FormatChild child) {
  auto result = node_name(node);
  if (node.children().empty()) return result;
  result += '(';
  for (std::size_t i = 0; i < node.children().size(); ++i) {
    if (i) result += ", ";
    result += child(node.children()[i]);
  }
  return result + ')';
}

inline std::string expression(const eggc::RecExpr<tepl::OpNode>& expr) {
  std::vector<std::string> terms;
  for (const auto& node : expr.nodes)
    terms.push_back(
        format_node(node, [&](eggc::Id id) { return terms.at(id); }));
  return terms.at(expr.root());
}

template <class Analysis>
void print_egraph(const eggc::EGraph<tepl::OpNode, Analysis>& graph,
                  eggc::Id root) {
  auto classes = graph.classes();
  std::sort(classes.begin(), classes.end());
  std::cout << "\nSaturated e-graph (nodes in each e-class are equivalent):\n";
  for (auto id : classes) {
    std::cout << 'e' << id << metadata(graph.analysis_data(id))
              << (id == graph.find(root) ? " (root)" : "") << '\n';
    std::vector<std::string> nodes;
    for (const auto& node : graph.nodes(id)) {
      auto text = format_node(node, [&](eggc::Id child) {
        return "e" + std::to_string(graph.find(child));
      });
      if (node.op() == tepl::Op::Input) text = "input " + text;
      nodes.push_back(std::move(text));
    }
    std::sort(nodes.begin(), nodes.end());
    for (const auto& node : nodes) std::cout << "  " << node << '\n';
  }
  std::cout << '\n';
}
}  // namespace utils
