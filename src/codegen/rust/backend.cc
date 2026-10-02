#include "src/codegen/rust/backend.h"

#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>

#include "src/codegen/rust/dialect_emitter.h"
#include "src/codegen/rust/names.h"
#include "src/codegen/rust/rule_emitter.h"
#include "src/codegen/rust/runtime_files.h"

namespace tepl::codegen::rust {
GenerationResult Backend::generate(const core::Program& program,
                                   const Options& options) const {
  GenerationResult result;
  const auto& package = options.package_name;
  if (package.empty() ||
      !((package[0] >= 'a' && package[0] <= 'z') ||
        (package[0] >= 'A' && package[0] <= 'Z')) ||
      package.find_first_not_of(
          "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
          std::string::npos) {
    result.diagnostics.push_back(
        {{},
         "package name must start with an ASCII letter and contain only "
         "letters, digits, '_' or '-'"});
    return result;
  }
  try {
    Names names(program);
    for (auto file : runtimeFiles()) {
      file.path.insert(4, "ir/");
      if (file.path == "src/ir/op_node.rs") {
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
        {"Cargo.toml", "[package]\nname = " + quote(package) +
                           "\nversion = \"0.1.0\"\nedition = "
                           "\"2024\"\n\n[dependencies]\negg = \"0.11.0\"\n"});
    result.files.push_back({"src/lib.rs",
                            "//! Generated TEPL dialects and checked egg "
                            "rewrites.\npub mod ir;\n"});
    result.files.push_back(
        {"src/ir/mod.rs",
         "pub mod dialects;\npub mod op_node;\npub mod types;\npub mod "
         "pattern;\npub mod rules;\npub use op_node::{Op, OpAttrs, OpNode, "
         "NodeError, Arity, DialectOp};\npub use types::DType;\n"});
    std::string dialect_index;
    for (const auto& dialect : names.dialects) {
      dialect_index += "pub mod " + dialect.module + ";\n";
      result.files.push_back({"src/ir/dialects/" + dialect.module + ".rs",
                              emitDialect(program, names, dialect)});
    }
    result.files.push_back({"src/ir/dialects/mod.rs", dialect_index});
    std::map<std::string, std::vector<const core::Rule*>> groups;
    std::map<std::string, std::string> owners;
    for (const auto& rule : program.rules) {
      std::filesystem::path source(rule.source_name);
      auto path = options.rules_root.empty()
                      ? source.filename()
                      : std::filesystem::absolute(source)
                            .lexically_normal()
                            .lexically_relative(
                                std::filesystem::absolute(options.rules_root)
                                    .lexically_normal());
      if (rule.source_name == "<input>") path = "input.tepl";
      path.replace_extension();
      std::string module;
      for (const auto& part : path) {
        if (!module.empty()) module += '/';
        module += moduleName(part.string());
      }
      auto [entry, inserted] = owners.emplace(module, rule.source_name);
      if (!inserted && entry->second != rule.source_name)
        throw std::invalid_argument("rule module name collision for '" +
                                    module + "'");
      groups[module].push_back(&rule);
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
      result.files.push_back({"src/ir/rules/" +
                                  (directory.empty() ? "" : directory + "/") +
                                  "mod.rs",
                              contents});
    }
    for (const auto& [module, rules] : groups) {
      std::string qualified;
      for (char c : module) qualified += c == '/' ? "::" : std::string(1, c);
      result.files.push_back({"src/ir/rules/" + module + ".rs",
                              emitRules(program, names, rules, qualified)});
    }
  } catch (const std::invalid_argument& error) {
    result.files.clear();
    result.diagnostics.push_back({{}, error.what()});
  }
  return result;
}
}  // namespace tepl::codegen::rust
