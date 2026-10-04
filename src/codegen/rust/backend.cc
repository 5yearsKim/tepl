#include "src/codegen/rust/backend.h"

#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>

#include "src/codegen/rust/analysis_emitter.h"
#include "src/codegen/rust/dialect_emitter.h"
#include "src/codegen/rust/names.h"
#include "src/codegen/rust/rule_emitter.h"
#include "src/codegen/rust/template_files.h"

namespace tepl::codegen::rust {
GenerationResult Backend::generate(const core::Program& program,
                                   const Options& options) const {
  GenerationResult result;
  try {
    const auto project = planProject(program, options.rules_root);
    Names names(program, project);
    for (auto file : templateFiles()) {
      file.path.erase(0, std::string("src/").size());
      if (file.path == "op_node.rs") {
        const std::string marker = "// @tepl:op-types";
        const auto position = file.contents.find(marker);
        if (position == std::string::npos)
          throw std::invalid_argument(
              "op_node template is missing its operation types marker");
        file.contents.replace(position, marker.size(), emitOpUnion(names));
      }
      result.files.push_back(std::move(file));
    }
    result.files.push_back(
        {"mod.rs",
         "pub mod analysis;\n"
         "pub mod dialects;\npub mod op_node;\npub mod types;\n"
         "pub mod pattern;\npub mod rules;\n"
         "pub use op_node::{Op, OpAttrs, OpNode, "
         "NodeError, Arity, DialectOp};\npub use types::DType;\n"});
    result.files.push_back(
        {"analysis/shape.rs", emitShapeInference(program, names)});
    result.files.push_back(
        {"analysis/dtype.rs", emitDTypeInference(program, names)});
    std::string dialect_index;
    for (const auto& dialect : names.dialects) {
      dialect_index += "pub mod " + dialect.module + ";\n";
      result.files.push_back({"dialects/" + dialect.module + ".rs",
                              emitDialect(program, names, dialect)});
    }
    result.files.push_back({"dialects/mod.rs", dialect_index});
    std::map<std::string, std::vector<const core::Rule*>> groups;
    std::map<std::string, std::string> owners;
    for (const auto& source : project.modules) {
      std::string module;
      for (const auto& part : source.path) {
        if (!module.empty()) module += '/';
        module += moduleName(part);
      }
      auto [entry, inserted] = owners.emplace(module, source.source);
      if (!inserted && entry->second != source.source)
        throw std::invalid_argument("rule module name collision for '" +
                                    module + "'");
      for (auto id : source.rules)
        groups[module].push_back(&program.rules.at(id.value));
    }
    std::map<std::string, std::set<std::string>> indexes;
    indexes[""];
    for (const auto& [module, rules] : groups) {
      std::filesystem::path path(module);
      auto parent = path.parent_path();
      indexes[parent.generic_string()].insert(path.filename().string());
      while (!parent.empty()) {
        indexes[parent.parent_path().generic_string()].insert(
            parent.filename().string());
        parent = parent.parent_path();
      }
    }
    for (const auto& [directory, children] : indexes) {
      if (groups.contains(directory))
        throw std::invalid_argument(
            "rule file/directory module collision for '" + directory + "'");
      std::string contents;
      for (const auto& child : children) contents += "pub mod " + child + ";\n";
      result.files.push_back(
          {"rules/" + (directory.empty() ? "" : directory + "/") + "mod.rs",
           contents});
    }
    for (const auto& [module, rules] : groups) {
      std::string qualified;
      std::string root_path = "super::super";
      for (char c : module) {
        if (c == '/') {
          qualified += "::";
          root_path += "::super";
        } else {
          qualified += c;
        }
      }
      result.files.push_back(
          {"rules/" + module + ".rs",
           emitRules(program, names, rules, qualified, root_path)});
    }
  } catch (const std::invalid_argument& error) {
    result.files.clear();
    result.diagnostics.push_back({{}, error.what()});
  }
  return result;
}
}  // namespace tepl::codegen::rust
