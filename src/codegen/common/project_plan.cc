#include "src/codegen/common/project_plan.h"

#include <filesystem>
#include <map>
#include <stdexcept>

namespace tepl::codegen {
ProjectPlan planProject(const executable::ProgramData& program,
                        const std::string& rules_root,
                        const std::string& graphs_root) {
  namespace fs = std::filesystem;
  const auto identity = [](const std::string& source) {
    return fs::absolute(source).lexically_normal().string();
  };
  const auto modulePath = [&](const std::string& source,
                              const std::string& root, const char* folder) {
    if (!root.empty())
      return fs::path(source).lexically_relative(
          fs::absolute(root).lexically_normal());
    auto parent = fs::path(source).parent_path();
    while (!parent.empty() && parent != parent.parent_path()) {
      if (parent.filename() == folder)
        return fs::path(source).lexically_relative(parent);
      parent = parent.parent_path();
    }
    return fs::path(source).filename();
  };
  ProjectPlan plan;
  std::map<std::pair<std::string, std::string>, std::size_t> dialect_ids;
  for (const auto& declaration : program.dialects) {
    auto source = identity(declaration.origin.definition.source_name);
    auto key = std::make_pair(source, declaration.name);
    if (!dialect_ids.emplace(key, plan.dialects.size()).second)
      throw std::invalid_argument("duplicate checked dialect declaration");
    plan.dialects.push_back({declaration.name, source, {}, {}});
  }
  for (const auto& op : program.operations)
    plan.dialects
        .at(dialect_ids.at(
            {identity(op.origin.definition.source_name), op.dialect}))
        .operations.push_back(op.id);
  for (const auto& schema : program.attribute_schemas)
    plan.dialects
        .at(dialect_ids.at(
            {identity(schema.origin.definition.source_name), schema.dialect}))
        .schemas.push_back(schema.id);

  std::map<std::vector<std::string>, ModulePlan> modules;
  for (const auto& rule : program.rules) {
    auto source = identity(rule.source_name);
    fs::path path = modulePath(source, rules_root, "rules");
    if (rule.source_name == "<input>" && rules_root.empty())
      path = "input.tepl";
    path.replace_extension();
    std::vector<std::string> parts;
    for (const auto& part : path) {
      if (part == ".." || part == "." || part.empty())
        throw std::invalid_argument("rule source is outside rules root: " +
                                    rule.source_name);
      parts.push_back(part.string());
    }
    if (parts.empty())
      throw std::invalid_argument("rule source has no module path: " +
                                  rule.source_name);
    auto [entry, inserted] =
        modules.try_emplace(parts, ModulePlan{source, parts, {}});
    if (!inserted && entry->second.source != source)
      throw std::invalid_argument("rule source module path collision: " +
                                  rule.source_name);
    entry->second.rules.push_back(rule.id);
  }
  for (auto& [path, module] : modules)
    plan.modules.push_back(std::move(module));
  std::map<std::vector<std::string>, GraphModulePlan> graph_modules;
  for (const auto& graph : program.graphs) {
    auto source = identity(graph.source_name);
    fs::path path = modulePath(source, graphs_root, "graphs");
    if (graph.source_name == "<input>" && graphs_root.empty())
      path = "input.tepl";
    path.replace_extension();
    std::vector<std::string> parts;
    for (const auto& part : path) {
      if (part == ".." || part == "." || part.empty())
        throw std::invalid_argument("graph source is outside graphs root: " +
                                    graph.source_name);
      parts.push_back(part.string());
    }
    if (parts.empty())
      throw std::invalid_argument("graph source has no module path");
    auto [entry, inserted] =
        graph_modules.try_emplace(parts, GraphModulePlan{source, parts, {}});
    if (!inserted && entry->second.source != source)
      throw std::invalid_argument("graph source module path collision");
    entry->second.graphs.push_back(graph.id);
  }
  for (auto& [path, module] : graph_modules)
    plan.graph_modules.push_back(std::move(module));
  return plan;
}
}  // namespace tepl::codegen
