#include "src/imports.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <unordered_set>
#include <utility>

#include "src/ast/ast.h"
#include "src/parse.h"

namespace tepl {

std::vector<Diagnostic> resolveImports(ast::Program& program) {
  namespace fs = std::filesystem;
  std::vector<Diagnostic> diagnostics;
  std::unordered_set<std::string> active;
  std::unordered_set<std::string> loaded;
  std::unordered_set<std::string> dialect_keys;
  std::unordered_set<std::string> rule_keys;
  std::unordered_set<std::string> scope_keys;
  const auto key = [](const ast::Dialect& dialect) {
    return dialect.source_name + "\n" + dialect.name;
  };
  for (const auto& dialect : program.dialects)
    dialect_keys.insert(key(dialect));
  for (const auto& rule : program.imported_rules)
    rule_keys.insert(rule.source_name + "\n" + rule.name);
  for (const auto& scope : program.imported_scopes)
    scope_keys.insert(scope.source_name);
  const auto root = fs::absolute(program.source_name).lexically_normal();
  active.insert(root.string());
  loaded.insert(root.string());

  std::function<void(const ast::Program&, const fs::path&)> load =
      [&](const ast::Program& current, const fs::path& file) {
        for (const auto& imported : current.imports) {
          const auto target = fs::absolute(file.parent_path() / imported.path)
                                  .lexically_normal();
          const auto report = [&](std::string message) {
            diagnostics.push_back({imported.span.begin.line,
                                   imported.span.begin.column,
                                   std::move(message), file.string()});
          };
          if (active.contains(target.string())) {
            report("cyclic import '" + imported.path + "'");
            continue;
          }
          std::ifstream input(target, std::ios::binary);
          if (!input) {
            report("cannot open import '" + imported.path + "'");
            continue;
          }
          std::string source{std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>()};
          if (input.bad()) {
            report("cannot read import '" + imported.path + "'");
            continue;
          }
          auto parsed = parse(source, target.string());
          if (!parsed.ok()) {
            for (const auto& diagnostic : parsed.diagnostics) {
              diagnostics.push_back({diagnostic.line, diagnostic.column,
                                     diagnostic.message, target.string()});
            }
            continue;
          }
          if (imported.rules.empty() && !parsed.program->rules.empty()) {
            report("import '" + imported.path +
                   "' contains rules; imports must define dialects only");
            continue;
          }
          std::unordered_set<std::string> direct_names;
          for (const auto& dialect : parsed.program->dialects) {
            if (!direct_names.insert(dialect.name).second) {
              diagnostics.push_back({dialect.span.begin.line,
                                     dialect.span.begin.column,
                                     "duplicate dialect '" + dialect.name + "'",
                                     target.string()});
            }
          }
          if (imported.dialect) {
            const bool found =
                std::any_of(parsed.program->dialects.begin(),
                            parsed.program->dialects.end(),
                            [&](const ast::Dialect& dialect) {
                              return dialect.name == *imported.dialect;
                            });
            if (!found) {
              report("dialect '" + *imported.dialect +
                     "' is not defined in import '" + imported.path + "'");
              continue;
            }
          }
          std::vector<const ast::Rule*> selected_rules;
          for (const auto& name : imported.rules) {
            const ast::Rule* selected = nullptr;
            bool duplicate = false;
            for (const auto& rule : parsed.program->rules) {
              if (rule.name != name.name) continue;
              if (selected) {
                report("duplicate rule '" + name.name + "' in import '" +
                       imported.path + "'");
                duplicate = true;
                break;
              }
              selected = &rule;
            }
            if (duplicate) continue;
            if (!selected) {
              report("rule '" + name.name + "' is not defined in import '" +
                     imported.path + "'");
            } else if (!selected->is_abstract) {
              report("imported rule '" + name.name + "' must be abstract");
            } else {
              selected_rules.push_back(selected);
            }
          }
          if (loaded.insert(target.string()).second) {
            if (scope_keys.insert(parsed.program->source_name).second)
              program.imported_scopes.push_back({parsed.program->source_name,
                                                 parsed.program->imports,
                                                 parsed.program->uses});
            active.insert(target.string());
            load(*parsed.program, target);
            active.erase(target.string());
          }
          for (const auto* rule : selected_rules) {
            if (rule_keys.insert(rule->source_name + "\n" + rule->name)
                    .second) {
              program.imported_rules.push_back(*rule);
            }
          }
          for (auto& dialect : parsed.program->dialects) {
            if (imported.dialect && dialect.name != *imported.dialect) continue;
            if (dialect_keys.insert(key(dialect)).second) {
              program.dialects.push_back(std::move(dialect));
            }
          }
        }
      };
  load(program, root);
  return diagnostics;
}

}  // namespace tepl
