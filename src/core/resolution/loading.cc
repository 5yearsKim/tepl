#include "src/core/resolution/loading.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <regex>
#include <set>

#include "src/core/syntax/parse.h"
#include "src/support/project_config.h"

namespace tepl {
namespace {
namespace fs = std::filesystem;
using Project = core::resolution::SourceProject;
std::string identity(const std::string& name) {
  return fs::absolute(name).lexically_normal().string();
}
// One compilation sees one parsed snapshot of every source, including roots.
class LoadingSession {
 public:
  explicit LoadingSession(SourceLoader& loader) : loader_(loader) {}
  void seed(const syntax::ParsedModule& module) {
    modules_.try_emplace(identity(module.source_name), module);
  }
  bool failed() const { return !diagnostics_.empty(); }
  // Cache configuration alongside modules so discovery and imports share it.
  std::optional<support::ProjectConfig> projectConfig(
      const std::string& source) {
    const auto directory = fs::path(source).parent_path().string();
    auto [root, unseen] = roots_.try_emplace(directory);
    if (unseen) {
      if (const auto path = loader_.findProjectConfig(source))
        root->second = identity(*path);
    }
    if (!root->second) return {};
    const auto& config_name = *root->second;
    auto [entry, added] = configs_.try_emplace(config_name);
    if (added) {
      if (const auto text = loader_.read(config_name)) {
        auto parsed = support::parseProjectConfig({config_name, *text});
        entry->second = std::move(parsed.config);
        diagnostics_.insert(diagnostics_.end(), parsed.diagnostics.begin(),
                            parsed.diagnostics.end());
      } else {
        diagnostics_.emplace_back(
            1, 1, "cannot read project config '" + config_name + "'",
            config_name);
      }
    }
    return entry->second;
  }
  LoadResult run(const std::vector<std::string>& roots, std::string name) {
    const auto config =
        name == "<project>" ? std::nullopt : projectConfig(identity(name));
    Project project;
    project.identity = std::move(name);
    project_ = &project;
    for (const auto& root : roots) {
      const auto* module = read(identity(root));
      if (!module) continue;
      addModule(*module);
      project.declarations.rules.insert(project.declarations.rules.end(),
                                        module->rules.begin(),
                                        module->rules.end());
      project.declarations.graphs.insert(project.declarations.graphs.end(),
                                         module->graphs.begin(),
                                         module->graphs.end());
      for (const auto& d : module->dialects) addDialect(d);
    }
    for (const auto& root : roots)
      if (const auto* module = read(identity(root))) visit(*module);
    std::set<std::string> local;
    for (const auto& r : project.declarations.rules) local.insert(key(r));
    std::erase_if(project.imported_rules,
                  [&](const auto& r) { return local.contains(key(r)); });
    LoadResult result;
    result.config = config;
    result.rule_count = project.declarations.rules.size();
    result.diagnostics = std::move(diagnostics_);
    if (result.diagnostics.empty()) result.program = std::move(project);
    return result;
  }

 private:
  template <class T>
  static std::string key(const T& d) {
    return d.source_name + "\n" + d.name;
  }
  const syntax::ParsedModule* read(const std::string& name,
                                   const syntax::Import* imported = nullptr,
                                   const std::string& source = {}) {
    if (auto it = modules_.find(name); it != modules_.end())
      return it->second ? &*it->second : nullptr;
    auto [it, added] = modules_.emplace(name, std::nullopt);
    auto text = loader_.read(name);
    if (!text) {
      if (imported)
        diagnostics_.push_back(
            {imported->span.begin.line, imported->span.begin.column,
             "cannot open import '" + imported->path + "'", source});
      else
        diagnostics_.push_back(
            {1, 1, "cannot read source '" + name + "'", name});
      return nullptr;
    }
    auto parsed = parse(*text, name);
    diagnostics_.insert(diagnostics_.end(), parsed.diagnostics.begin(),
                        parsed.diagnostics.end());
    if (parsed.ok()) it->second = std::move(*parsed.program);
    return it->second ? &*it->second : nullptr;
  }
  void addModule(const syntax::ParsedModule& module) {
    if (included_.insert(identity(module.source_name)).second)
      project_->modules.push_back(module);
  }
  void addDialect(const syntax::Dialect& d) {
    if (dialects_.insert(key(d)).second)
      project_->declarations.dialects.push_back(d);
  }
  std::optional<std::string> projectRoot(const std::string& source) {
    if (const auto config = projectConfig(source)) return config->root;
    return {};
  }
  void visit(const syntax::ParsedModule& module) {
    const auto name = identity(module.source_name);
    if (complete_.contains(name)) return;
    const auto root = projectRoot(name);
    active_.insert(name);
    for (const auto& imported : module.imports) {
      const fs::path path(imported.path);
      const bool relative =
          !path.empty() && (*path.begin() == "." || *path.begin() == "..");
      const auto base =
          root && !relative ? fs::path(*root) : fs::path(name).parent_path();
      const auto target = identity((base / path).string());
      project_->import_targets.emplace(std::pair{name, imported.path}, target);
      auto report = [&](std::string message) {
        diagnostics_.push_back({imported.span.begin.line,
                                imported.span.begin.column, std::move(message),
                                module.source_name});
      };
      if (active_.contains(target)) {
        report("cyclic import '" + imported.path + "'");
        continue;
      }
      const auto* dependency = read(target, &imported, module.source_name);
      if (!dependency) continue;
      if (imported.rules.empty() &&
          (!dependency->rules.empty() || !dependency->graphs.empty())) {
        report("import '" + imported.path +
               "' contains rules or graphs; imports must define dialects only");
        continue;
      }
      std::set<std::string> names;
      for (const auto& d : dependency->dialects)
        if (!names.insert(d.name).second)
          report("duplicate dialect '" + d.name + "'");
      if (imported.dialect && !names.contains(*imported.dialect)) {
        report("dialect '" + *imported.dialect +
               "' is not defined in import '" + imported.path + "'");
        continue;
      }
      for (const auto& selected : imported.rules) {
        const syntax::Rule* rule = nullptr;
        bool duplicate = false;
        for (const auto& r : dependency->rules)
          if (r.name == selected.name) {
            if (rule) {
              duplicate = true;
              break;
            }
            rule = &r;
          }
        if (duplicate)
          report("duplicate rule '" + selected.name + "' in import '" +
                 imported.path + "'");
        else if (!rule)
          report("rule '" + selected.name + "' is not defined in import '" +
                 imported.path + "'");
        else if (!rule->is_abstract)
          report("imported rule '" + selected.name + "' must be abstract");
        else if (rules_.insert(key(*rule)).second)
          project_->imported_rules.push_back(*rule);
      }
      addModule(*dependency);
      visit(*dependency);
      for (const auto& d : dependency->dialects)
        if (!imported.dialect || d.name == *imported.dialect) addDialect(d);
    }
    active_.erase(name);
    complete_.insert(name);
  }
  SourceLoader& loader_;
  Project* project_ = nullptr;
  std::map<std::string, std::optional<syntax::ParsedModule>> modules_;
  std::map<std::string, std::optional<std::string>> roots_;
  std::map<std::string, std::optional<support::ProjectConfig>> configs_;
  std::set<std::string> active_, complete_, included_, dialects_, rules_;
  std::vector<Diagnostic> diagnostics_;
};
// Match project-relative paths: * and ? stay within a component, ** crosses
// directories, and **/ also matches zero directory components.
std::regex exclusionPattern(const std::string& glob) {
  std::string pattern = "^";
  for (std::size_t i = 0; i < glob.size(); ++i) {
    const char ch = glob[i];
    if (ch == '*') {
      if (i + 1 < glob.size() && glob[i + 1] == '*') {
        ++i;
        if (i + 1 < glob.size() && glob[i + 1] == '/') {
          ++i;
          pattern += "(.*/)?";
        } else {
          pattern += ".*";
        }
      } else {
        pattern += "[^/]*";
      }
    } else if (ch == '?') {
      pattern += "[^/]";
    } else {
      if (std::string_view(".\\+()[]{}^$|").find(ch) != std::string_view::npos)
        pattern += '\\';
      pattern += ch;
    }
  }
  return std::regex(pattern + "$", std::regex::ECMAScript);
}
void collectFiles(const fs::path& path, std::vector<std::string>& files,
                  const std::optional<support::ProjectConfig>& config = {}) {
  if (fs::is_directory(path)) {
    std::vector<std::regex> excludes;
    if (config)
      for (const auto& pattern : config->sources.exclude)
        excludes.push_back(exclusionPattern(pattern));
    for (const auto& entry : fs::recursive_directory_iterator(path)) {
      if (!entry.is_regular_file() || entry.path().extension() != ".tepl")
        continue;
      const auto name = identity(entry.path().string());
      const auto relative =
          config
              ? fs::path(name).lexically_relative(config->root).generic_string()
              : name;
      if (std::none_of(excludes.begin(), excludes.end(),
                       [&](const auto& pattern) {
                         return std::regex_match(relative, pattern);
                       }))
        files.push_back(name);
    }
  } else {
    // Explicit file inputs are intentional, even when excluded from discovery.
    files.push_back(identity(path.string()));
  }
}
void collectDirectories(const fs::path& root,
                        const std::vector<std::string>& directories,
                        const std::optional<support::ProjectConfig>& config,
                        std::vector<std::string>& files) {
  for (const auto& directory : directories) {
    const auto path = root / directory;
    if (fs::is_directory(path)) collectFiles(path, files, config);
  }
}
void sortFiles(std::vector<std::string>& files) {
  std::sort(files.begin(), files.end());
  files.erase(std::unique(files.begin(), files.end()), files.end());
}
}  // namespace
std::optional<std::string> FileSourceLoader::read(const std::string& name) {
  std::ifstream input(name, std::ios::binary);
  if (!input) return {};
  try {
    std::string text{std::istreambuf_iterator<char>(input), {}};
    if (input.bad()) return {};
    return text;
  } catch (const std::ios_base::failure&) {
    return {};
  }
}
std::optional<std::string> FileSourceLoader::findProjectConfig(
    const std::string& source) {
  auto directory = fs::path(identity(source)).parent_path();
  for (;;) {
    const auto config = directory / "tepl.toml";
    std::error_code error;
    const auto status = fs::symlink_status(config, error);
    if (error && error != std::errc::no_such_file_or_directory)
      throw fs::filesystem_error("cannot inspect project config", config,
                                 error);
    if (fs::exists(status)) return config.string();
    const auto parent = directory.parent_path();
    if (parent == directory) return {};
    directory = parent;
  }
}
LoadResult loadImports(const syntax::ParsedModule& module,
                       SourceLoader& loader) {
  LoadingSession session(loader);
  session.seed(module);
  return session.run({module.source_name}, module.source_name);
}
LoadResult loadImports(const syntax::ParsedModule& module) {
  FileSourceLoader loader;
  return loadImports(module, loader);
}
LoadResult loadProject(const std::vector<SourceFile>& roots,
                       SourceLoader& loader) {
  LoadingSession session(loader);
  std::vector<std::string> names;
  std::vector<Diagnostic> diagnostics;
  for (const auto& source : roots) {
    auto parsed = parse(SourceFile{identity(source.identity), source.text});
    diagnostics.insert(diagnostics.end(), parsed.diagnostics.begin(),
                       parsed.diagnostics.end());
    if (parsed.ok()) {
      session.seed(*parsed.program);
      names.push_back(parsed.program->source_name);
    }
  }
  auto result = session.run(names, "<project>");
  result.diagnostics.insert(result.diagnostics.end(), diagnostics.begin(),
                            diagnostics.end());
  if (!result.diagnostics.empty()) result.program.reset();
  return result;
}
LoadResult loadProject(const std::string& directory, SourceLoader& loader) {
  const auto root = fs::path(identity(directory));
  LoadingSession session(loader);
  const auto name = (root / "<project>").string();
  const auto config = session.projectConfig(name);
  if (session.failed()) return session.run({}, name);
  const auto sources = config ? config->sources : support::ProjectSources{};
  // Configured paths are relative to the config, even for a nested input
  // directory.
  const auto base = config ? fs::path(config->root) : root;
  std::vector<std::string> files;
  for (const auto& directories :
       {sources.dialects, sources.rules, sources.graphs})
    collectDirectories(base, directories, config, files);
  sortFiles(files);
  auto result = session.run(files, name);
  if (files.empty() && result.diagnostics.empty()) {
    result.diagnostics.push_back(
        {1, 1,
         "no .tepl files found in configured source directories (defaults: "
         "dialects/, rules/, graphs/)",
         directory});
    result.program.reset();
  }
  return result;
}
LoadResult loadProject(const std::string& directory) {
  FileSourceLoader loader;
  return loadProject(directory, loader);
}
LoadResult loadRewriteSources(const std::string& graph,
                              const std::string& rules, SourceLoader& loader) {
  const auto input = fs::path(identity(graph));
  LoadingSession session(loader);
  const bool directory = fs::is_directory(input);
  const auto name = directory ? (input / "<project>").string() : input.string();
  const auto config = session.projectConfig(name);
  if (session.failed()) return session.run({}, name);
  std::vector<std::string> files, rule_files;
  if (directory && ((config && fs::path(config->root) == input) ||
                    fs::is_directory(input / "graphs") ||
                    fs::is_directory(input / "dialects") ||
                    fs::is_directory(input / "rules"))) {
    const auto sources = config ? config->sources : support::ProjectSources{};
    collectDirectories(input, sources.dialects, config, files);
    collectDirectories(input, sources.graphs, config, files);
  } else {
    collectFiles(input, files, config);
  }
  const auto rule_path = fs::path(identity(rules));
  const auto rule_config = session.projectConfig(
      fs::is_directory(rule_path) ? (rule_path / "<project>").string()
                                  : rule_path.string());
  if (session.failed()) return session.run({}, name);
  collectFiles(rule_path, rule_files, rule_config);
  const bool graph_empty = files.empty();
  files.insert(files.end(), rule_files.begin(), rule_files.end());
  sortFiles(files);
  auto result = session.run(files, name);
  for (const auto& [path, empty] :
       {std::pair{graph, graph_empty}, std::pair{rules, rule_files.empty()}})
    if (empty) {
      result.diagnostics.push_back(
          {1, 1, "no .tepl files found in selected source directory", path});
      result.program.reset();
    }
  return result;
}
LoadResult loadRewriteSources(const std::string& graph,
                              const std::string& rules) {
  FileSourceLoader loader;
  return loadRewriteSources(graph, rules, loader);
}
}  // namespace tepl
