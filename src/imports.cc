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
          if (imported.rules.empty() && (!parsed.program->rules.empty() ||
                                         !parsed.program->graphs.empty())) {
            report("import '" + imported.path +
                   "' contains rules or graphs; imports must define dialects "
                   "only");
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

namespace tepl {
namespace {
ParseResult loadFiles(const std::vector<std::filesystem::path>& files,
                      const std::filesystem::path& root) {
  namespace fs = std::filesystem;
  ParseResult result;
  result.program.emplace();
  auto& project = *result.program;
  project.source_name = (root / "<project>").string();
  std::unordered_set<std::string> dialects, scopes, instances, imported;
  for (const auto& file : files) {
    std::ifstream input(file, std::ios::binary);
    if (!input) {
      result.diagnostics.push_back(
          {1, 1, "cannot read project file", file.string()});
      continue;
    }
    std::string source{std::istreambuf_iterator<char>(input), {}};
    if (input.bad()) {
      result.diagnostics.push_back(
          {1, 1, "cannot read project file", file.string()});
      continue;
    }
    auto parsed = parse(source, file.string());
    if (parsed.ok()) {
      auto diagnostics = resolveImports(*parsed.program);
      parsed.diagnostics.insert(parsed.diagnostics.end(), diagnostics.begin(),
                                diagnostics.end());
    }
    result.diagnostics.insert(result.diagnostics.end(),
                              parsed.diagnostics.begin(),
                              parsed.diagnostics.end());
    if (!parsed.ok()) continue;
    auto& module = *parsed.program;
    if (scopes.insert(module.source_name).second)
      project.imported_scopes.push_back(
          {module.source_name, module.imports, module.uses});
    for (auto& scope : module.imported_scopes)
      if (scopes.insert(scope.source_name).second)
        project.imported_scopes.push_back(std::move(scope));
    for (auto& dialect : module.dialects)
      if (dialects.insert(dialect.source_name + "\n" + dialect.name).second)
        project.dialects.push_back(std::move(dialect));
    for (auto& graph : module.graphs)
      project.graphs.push_back(std::move(graph));
    for (auto& rule : module.rules) {
      instances.insert(rule.source_name + "\n" + rule.name);
      project.rules.push_back(std::move(rule));
    }
    for (auto& rule : module.imported_rules)
      if (imported.insert(rule.source_name + "\n" + rule.name).second)
        project.imported_rules.push_back(std::move(rule));
  }
  std::erase_if(project.imported_rules, [&](const ast::Rule& rule) {
    return instances.contains(rule.source_name + "\n" + rule.name);
  });
  result.rule_count = project.rules.size();
  return result;
}

void collectFiles(const std::filesystem::path& path,
                  std::vector<std::filesystem::path>& files) {
  namespace fs = std::filesystem;
  if (fs::is_directory(path)) {
    std::vector<fs::path> found;
    for (const auto& entry : fs::recursive_directory_iterator(path))
      if (entry.is_regular_file() && entry.path().extension() == ".tepl")
        found.push_back(entry.path());
    std::sort(found.begin(), found.end());
    files.insert(files.end(), found.begin(), found.end());
  } else {
    files.push_back(path);
  }
}
}  // namespace

ParseResult loadProject(const std::string& directory) {
  namespace fs = std::filesystem;
  const auto root = fs::absolute(directory).lexically_normal();
  std::vector<fs::path> files;
  for (const auto* part : {"dialects", "rules", "graphs"})
    if (fs::is_directory(root / part)) collectFiles(root / part, files);
  std::sort(files.begin(), files.end());
  auto result = loadFiles(files, root);
  if (files.empty())
    result.diagnostics.push_back(
        {1, 1, "no .tepl files found under dialects/, rules/, or graphs/",
         directory});
  return result;
}

ParseResult loadRewriteSources(const std::string& graph,
                               const std::string& rules) {
  namespace fs = std::filesystem;
  const auto input = fs::absolute(graph).lexically_normal();
  const auto ruleInput = fs::absolute(rules).lexically_normal();
  std::vector<fs::path> graphFiles, ruleFiles;
  if (fs::is_directory(input) && (fs::is_directory(input / "graphs") ||
                                  fs::is_directory(input / "dialects") ||
                                  fs::is_directory(input / "rules"))) {
    for (const auto* part : {"dialects", "graphs"})
      if (fs::is_directory(input / part))
        collectFiles(input / part, graphFiles);
  } else {
    collectFiles(input, graphFiles);
  }
  collectFiles(ruleInput, ruleFiles);
  auto files = graphFiles;
  files.insert(files.end(), ruleFiles.begin(), ruleFiles.end());
  std::unordered_set<std::string> seen;
  std::erase_if(files, [&](const auto& file) {
    return !seen.insert(fs::weakly_canonical(file).string()).second;
  });
  auto result =
      loadFiles(files, fs::is_directory(input) ? input : input.parent_path());
  for (const auto& [path, empty] : {std::pair{input, graphFiles.empty()},
                                    std::pair{ruleInput, ruleFiles.empty()}})
    if (empty)
      result.diagnostics.push_back(
          {1, 1, "no .tepl files found in selected source directory",
           path.string()});
  return result;
}
}  // namespace tepl
