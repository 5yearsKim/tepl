#include "src/core/syntax/print.h"

#include "src/core/resolution/loading.h"
namespace tepl {
std::string formatAst(const core::resolution::SourceProject& project) {
  syntax::ParsedModule declarations;
  declarations.source_name = project.identity;
  if (!project.modules.empty() &&
      project.modules.front().source_name == project.identity) {
    declarations.span = project.modules.front().span;
    declarations.imports = project.modules.front().imports;
    declarations.uses = project.modules.front().uses;
  }
  declarations.rules = project.declarations.rules;
  declarations.dialects = project.declarations.dialects;
  declarations.graphs = project.declarations.graphs;
  auto text = formatAst(declarations);
  for (const auto& rule : project.imported_rules) {
    syntax::ParsedModule module;
    module.source_name = rule.source_name;
    module.rules.push_back(rule);
    text += "  imported from " + rule.source_name + "\n" + formatAst(module);
  }
  return text;
}
}  // namespace tepl
