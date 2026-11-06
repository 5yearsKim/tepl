#pragma once

#include <memory>

namespace tepl::core::semantics {
// Read-only preorder over arguments. Access maps and metadata parameters are
// separate trees; callers choose explicitly when to visit them.
template <class Node, class Visitor>
void walkTree(const std::shared_ptr<const Node>& node, const Visitor& visit) {
  visit(node);
  for (const auto& arg : node->args) walkTree(arg, visit);
}

// Preserve every node field while replacing arguments. Scope checks belong in
// the callback; a rejected child rejects the whole transformation.
template <class Node, class Transform>
std::shared_ptr<const Node> mapArguments(
    const std::shared_ptr<const Node>& node, const Transform& transform) {
  auto result = std::make_shared<Node>(*node);
  for (auto& arg : result->args) {
    arg = transform(arg);
    if (!arg) return {};
  }
  return result;
}
}  // namespace tepl::core::semantics
