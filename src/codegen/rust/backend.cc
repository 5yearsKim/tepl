#include "src/codegen/rust/backend.h"

#include <stdexcept>

#include "src/codegen/rust/analysis_emitter.h"
#include "src/codegen/rust/dialect_emitter.h"
#include "src/codegen/rust/graph_emitter.h"
#include "src/codegen/rust/names.h"
#include "src/codegen/rust/rule_emitter.h"
#include "src/codegen/rust/template_files.h"

namespace tepl::codegen::rust {
GenerationResult Backend::generate(const core::Program& program,
                                   const Options& options) const {
  GenerationResult result;
  try {
    const auto project =
        planProject(program, options.rules_root, options.graphs_root);
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
      if (file.path != "graphs/mod.rs") result.files.push_back(std::move(file));
    }
    result.files.push_back(
        {"mod.rs",
         "pub mod analysis;\npub mod builtins;\n"
         "pub mod dialects;\npub mod op_node;\npub mod types;\n"
         "pub mod rewriting;\npub mod rules;\npub mod graphs;\n"
         "pub use op_node::{Op, OpAttrs, OpNode, "
         "NodeError, Arity, DialectOp};\npub use types::DType;\n"
         "pub use analysis::{Inference, TensorAnalysis, TensorAnalysisData, "
         "TensorBindingTable, TensorInfo};\n"
         "pub use graphs::{BuiltGraph, GraphDefinition, InputSpec};\n"
         "pub use rewriting::RewriteAnalysis;\n"});
    result.files.push_back(
        {"analysis/shape.rs", emitShapeInference(program, names)});
    result.files.push_back(
        {"analysis/dtype.rs", emitDTypeInference(program, names)});
    std::string dialect_index;
    for (const auto& dialect : names.dialects) {
      dialect_index += "pub mod " + dialect.module + ";\n";
      result.files.push_back({"dialects/" + dialect.file_stem + ".rs",
                              emitDialect(program, names, dialect)});
    }
    result.files.push_back({"dialects/mod.rs", dialect_index});
    for (const auto& [directory, children] : names.module_indexes) {
      std::string contents;
      for (const auto& child : children) contents += "pub mod " + child + ";\n";
      result.files.push_back(
          {"rules/" + (directory.empty() ? "" : directory + "/") + "mod.rs",
           contents});
    }
    for (const auto& module : names.modules) {
      std::vector<const core::Rule*> rules;
      for (auto id : module.rules) rules.push_back(&program.rules.at(id.value));
      result.files.push_back(
          {"rules/" + module.path + ".rs",
           emitRules(program, names, rules, module.qualified, module.root)});
    }
    emitGraphs(program, names, project, result);
  } catch (const NameError& error) {
    result.files.clear();
    result.diagnostics.push_back({error.origin, error.what()});
  } catch (const std::invalid_argument& error) {
    result.files.clear();
    result.diagnostics.push_back({{}, error.what()});
  }
  return result;
}
}  // namespace tepl::codegen::rust
