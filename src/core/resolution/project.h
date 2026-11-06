#pragma once
#include <map>
#include <string>
#include <utility>

#include "src/core/syntax/ast.h"
namespace tepl::core::resolution {
struct ProjectDeclarations {
  std::vector<syntax::Rule> rules;
  std::vector<syntax::Dialect> dialects;
  std::vector<syntax::ConcreteGraph> graphs;
};
struct SourceProject {
  std::string identity;
  // Complete modules retain their own imports, uses, and source identity.
  std::vector<syntax::ParsedModule> modules;
  // Selected declarations used by this compilation.
  ProjectDeclarations declarations;
  std::vector<syntax::Rule> imported_rules;
  // (normalized importing source, written import path) -> normalized target.
  // Later stages consume these edges without filesystem or config discovery.
  std::map<std::pair<std::string, std::string>, std::string> import_targets;
};
}  // namespace tepl::core::resolution
