#include "src/core/resolution/file_scope.h"

#include <map>
#include <set>

#include "src/core/analysis_context.h"
#include "src/core/resolution/source.h"

namespace tepl::core::detail {

namespace {

using Aliases = std::map<std::string, DeclarationKey>;
using Module = ast::Program::ImportedScope;

class FileScopeBuilder {
 public:
  FileScopeBuilder(AnalysisContext& context, const DialectRegistry& dialects,
                   const RuleRegistry& rules)
      : context_(context),
        root_{context.input.source_name, context.input.imports,
              context.input.uses},
        dialects_(dialects),
        rules_(rules) {}

  void run() {
    modules_.emplace(sourceKey(root_.source_name), &root_);
    for (const auto& module : context_.input.imported_scopes)
      modules_.emplace(sourceKey(module.source_name), &module);
    buildScope(root_, origin(root_.source_name, context_.input.span));
    for (const auto& module : context_.input.imported_scopes)
      buildScope(module, origin(module.source_name, {}));
  }

 private:
  void open(FileScope& scope, const std::string& name, OpId id,
            const SourceOrigin& at) {
    const auto [found, inserted] = scope.operations.emplace(name, id);
    if (!inserted && found->second != id)
      context_.report(at, "ambiguous operation '" + name + "'");
  }

  void openAll(FileScope& scope, const DeclarationKey& dialect,
               const SourceOrigin& at) {
    for (const auto& [name, id] : dialects_.at(dialect).operations)
      open(scope, name, id, at);
  }

  void alias(FileScope& scope, Aliases& aliases, const std::string& name,
             const DeclarationKey& dialect, const SourceOrigin& at) {
    const auto [found, inserted] = aliases.emplace(name, dialect);
    if (!inserted && found->second != dialect) {
      context_.report(at, "duplicate dialect alias '" + name + "'");
      return;
    }
    for (const auto& [operation, id] : dialects_.at(dialect).operations)
      open(scope, name + "." + operation, id, at);
  }

  void openImport(FileScope& scope, Aliases& aliases, const std::string& source,
                  const ast::Import& imported) {
    const auto at = origin(source, imported.span);
    const auto target = importTarget(source, imported.path);
    const auto module = modules_.find(target);
    if (module == modules_.end()) {
      context_.report(
          at, "import '" + imported.path +
                  "' is not loaded; call resolveImports before analysis");
      return;
    }
    if (!imported.rules.empty()) {
      for (const auto& name : imported.rules) {
        const auto found = rules_.find({target, name.name});
        if (found == rules_.end()) {
          context_.report(at,
                          "imported rule '" + name.name + "' is not loaded");
          continue;
        }
        const auto [previous, inserted] =
            scope.templates.emplace(name.name, found->second);
        if (!inserted && previous->second != found->second)
          context_.report(at, "ambiguous imported rule '" + name.name + "'");
      }
      return;
    }
    if (imported.dialect) {
      const DeclarationKey key{target, *imported.dialect};
      if (!dialects_.contains(key)) {
        context_.report(
            at, "imported dialect '" + *imported.dialect + "' is not loaded");
        return;
      }
      alias(scope, aliases, imported.alias.value_or(*imported.dialect), key,
            at);
      return;
    }
    if (!buildScope(*module->second, at)) return;
    // A legacy import exports bare names, including nested legacy imports.
    for (const auto& [name, id] : context_.scopes.at(target).operations)
      if (name.find('.') == std::string::npos) open(scope, name, id, at);
  }

  void openUse(FileScope& scope, const Aliases& aliases,
               const std::string& source, const ast::Use& used) {
    const auto at = origin(source, used.span);
    const auto found = aliases.find(used.alias);
    if (found == aliases.end()) {
      context_.report(at, "unknown dialect alias '" + used.alias + "'");
      return;
    }
    if (used.operations.empty()) {
      openAll(scope, found->second, at);
      return;
    }
    const auto& names = dialects_.at(found->second).operations;
    for (const auto& name : used.operations) {
      const auto operation = names.find(name);
      if (operation == names.end())
        context_.report(at, "unknown operation '" + name + "' in dialect '" +
                                used.alias + "'");
      else
        open(scope, name, operation->second, at);
    }
  }

  bool buildScope(const Module& module, const SourceOrigin& at) {
    const auto key = sourceKey(module.source_name);
    if (complete_.contains(key)) return true;
    if (!active_.insert(key).second) {
      context_.report(at, "cyclic import scope");
      return false;
    }
    auto& scope = context_.scopes.try_emplace(key).first->second;
    context_.output.sources.push_back(module.source_name);
    Aliases aliases;
    for (const auto& [name, dialect] : dialects_) {
      if (name.first != key) continue;
      const auto location =
          origin(module.source_name, dialect.declaration->span);
      alias(scope, aliases, name.second, name, location);
      openAll(scope, name, location);
    }
    for (const auto& [name, rule] : rules_)
      if (name.first == key) scope.templates.emplace(name.second, rule);
    for (const auto& imported : module.imports)
      openImport(scope, aliases, module.source_name, imported);
    for (const auto& used : module.uses)
      openUse(scope, aliases, module.source_name, used);
    active_.erase(key);
    complete_.insert(key);
    return true;
  }
  AnalysisContext& context_;
  Module root_;
  const DialectRegistry& dialects_;
  const RuleRegistry& rules_;
  std::map<std::string, const Module*> modules_;
  std::set<std::string> active_;
  std::set<std::string> complete_;
};

}  // namespace

void buildFileScopes(AnalysisContext& context, const DialectRegistry& dialects,
                     const RuleRegistry& rules) {
  FileScopeBuilder(context, dialects, rules).run();
}

}  // namespace tepl::core::detail
