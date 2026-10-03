#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "src/core/analysis_context.h"
#include "src/core/resolution/source.h"
#include "src/core/shape/type_inference.h"

namespace tepl::core::shape::detail {

struct CheckContext {
  CheckContext(core::detail::AnalysisContext& analysis,
               const Operation& operation, std::string source)
      : analysis(analysis),
        operation(operation),
        source(std::move(source)),
        types(analysis.diagnostics) {}
  SourceOrigin origin(SourceSpan span) const {
    return core::detail::origin(source, span);
  }
  void report(const SourceOrigin& origin, std::string message) {
    analysis.report(origin, std::move(message));
  }
  SymbolId symbol(std::string name, TypeId type, SourceOrigin origin) {
    const SymbolId id{program.symbols.size()};
    program.symbols.push_back({id, std::move(name), type, std::move(origin)});
    return id;
  }
  bool bind(const std::string& name, SymbolId id) {
    if (name == "attrs") {
      report(program.symbols[id.value].origin,
             "shape binding cannot use reserved name 'attrs'");
      return false;
    }
    const auto [previous, inserted] = scope.emplace(name, id);
    if (!inserted)
      analysis.report(
          program.symbols[id.value].origin,
          "duplicate shape binding '" + name + "'",
          {program.symbols[previous->second.value].origin.definition});
    return inserted;
  }

  core::detail::AnalysisContext& analysis;
  const Operation& operation;
  std::string source;
  Program program;
  TypeInference types;
  std::unordered_map<std::string, SymbolId> scope;
};

}  // namespace tepl::core::shape::detail
