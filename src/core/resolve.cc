#include <map>
#include <set>
#include <utility>

#include "src/core/analysis_context.h"

namespace tepl::core::detail {
namespace {

// File identity and declared name are separate keys; no delimiter encoding.
using DeclarationKey = std::pair<std::string, std::string>;
using OperationNames = std::map<std::string, OpId>;
using SchemaNames = std::map<std::string, AttributeSchemaId>;
using Aliases = std::map<std::string, DeclarationKey>;
using Module = ast::Program::ImportedScope;

struct DialectSymbols {
  const ast::Dialect* declaration;
  OperationNames operations;
};

class Resolver {
 public:
  explicit Resolver(AnalysisContext& context)
      : context_(context),
        root_{context.input.source_name, context.input.imports,
              context.input.uses},
        tensor_(context.types.concrete({TypeKind::kTensor})) {}

  void run() {
    for (const auto& dialect : context_.input.dialects)
      registerDialect(dialect);
    for (const auto& rule : context_.input.rules) registerRule(rule);
    for (const auto& rule : context_.input.imported_rules) registerRule(rule);
    modules_.emplace(sourceKey(root_.source_name), &root_);
    for (const auto& module : context_.input.imported_scopes)
      modules_.emplace(sourceKey(module.source_name), &module);
    buildScope(root_, origin(root_.source_name, context_.input.span));
    for (const auto& module : context_.input.imported_scopes)
      buildScope(module, origin(module.source_name, {}));
  }

 private:
  std::vector<AttributeField> checkFields(
      const std::vector<ast::AttrField>& fields, const std::string& source) {
    std::vector<AttributeField> result;
    std::set<std::string> names;
    for (const auto& field : fields) {
      const auto at = origin(source, field.span);
      if (!names.insert(field.name).second)
        context_.report(at, "duplicate attribute field '" + field.name + "'");
      if (field.type != "index" && field.type != "string")
        context_.report(at, "unknown attribute type '" + field.type + "'");
      if (field.empty_default && !field.list)
        context_.report(at, "empty list default requires a list type");
      result.push_back({field.name,
                        field.type == "index" ? AttributeField::Kind::kIndex
                                              : AttributeField::Kind::kString,
                        field.list, field.empty_default, at});
    }
    return result;
  }

  AttributeSchemaId addSchema(const ast::Dialect& dialect,
                              const std::string& name,
                              const std::vector<ast::AttrField>& fields,
                              const SourceOrigin& at) {
    auto& schemas = context_.output.attribute_schemas;
    const AttributeSchemaId id{schemas.size()};
    schemas.push_back(
        {id, dialect.name, name, checkFields(fields, dialect.source_name), at});
    return id;
  }

  void registerDialect(const ast::Dialect& dialect) {
    const DeclarationKey key{sourceKey(dialect.source_name), dialect.name};
    const auto [entry, inserted] =
        dialects_.emplace(key, DialectSymbols{&dialect, {}});
    if (!inserted) {
      context_.report(origin(dialect.source_name, dialect.span),
                      "duplicate dialect '" + dialect.name + "'");
      return;
    }
    SchemaNames schemas;
    for (const auto& schema : dialect.schemas) {
      const auto at = origin(dialect.source_name, schema.span);
      const auto id = addSchema(dialect, schema.name, schema.fields, at);
      if (!schemas.emplace(schema.name, id).second)
        context_.report(at, "duplicate attribute schema '" + schema.name + "'");
    }
    for (const auto& op : dialect.operations)
      registerOperation(dialect, op, schemas, entry->second.operations);
  }

  void registerOperation(const ast::Dialect& dialect, const ast::OpDecl& op,
                         const SchemaNames& schemas, OperationNames& names) {
    const auto at = origin(dialect.source_name, op.span);
    const OpId id{context_.output.operations.size()};
    Operation value{id, dialect.name, op.name,      op.alias,
                    {}, tensor_,      std::nullopt, at};
    if (op.result_type != "tensor")
      context_.report(
          at, "unsupported operation result type '" + op.result_type + "'");
    std::set<std::string> operands;
    for (std::size_t i = 0; i < op.operands.size(); ++i) {
      const auto& operand = op.operands[i];
      const auto location = origin(dialect.source_name, operand.span);
      if (!operands.insert(operand.name).second)
        context_.report(location, "duplicate operand '" + operand.name + "'");
      if (operand.type != "tensor")
        context_.report(location, "unsupported operation operand type '" +
                                      operand.type + "'");
      if (operand.variadic && i + 1 != op.operands.size())
        context_.report(location, "variadic operand must be last");
      value.operands.push_back(
          {operand.name, tensor_, operand.variadic, location});
    }
    if (op.attrs) {
      if (const auto* shared = std::get_if<ast::SharedAttrs>(&*op.attrs)) {
        const auto found = schemas.find(shared->name);
        if (found == schemas.end())
          context_.report(at,
                          "unknown attribute schema '" + shared->name + "'");
        else
          value.attributes = found->second;
      } else {
        const auto& fields = std::get<ast::InlineAttrs>(*op.attrs).fields;
        if (!fields.empty())
          value.attributes =
              addSchema(dialect, op.name + "::attrs", fields, at);
      }
    }
    context_.output.operations.push_back(std::move(value));
    const auto addName = [&](const std::string& name) {
      if (!names.emplace(name, id).second)
        context_.report(at, "duplicate operation name or alias '" + name + "'");
    };
    addName(op.name);
    if (op.alias) addName(*op.alias);
  }

  void checkSignatureType(const ast::SignatureType& type,
                          ast::RuleParameterKind kind,
                          const std::string& source) {
    const auto resolved = resolveType(type.name);
    if (!resolved || (kind == ast::RuleParameterKind::kOperation &&
                      resolved->kind != TypeKind::kTensor))
      context_.report(
          origin(source, type.span),
          "unsupported parameter signature type '" + type.name + "'");
  }

  void registerRule(const ast::Rule& rule) {
    if (!rules_
             .emplace(DeclarationKey{sourceKey(rule.source_name), rule.name},
                      &rule)
             .second)
      context_.report(origin(rule.source_name, rule.span),
                      "duplicate rule '" + rule.name + "'");
    std::set<std::string> names;
    for (const auto& parameter : rule.parameters) {
      if (!names.insert(parameter.name).second)
        context_.report(origin(rule.source_name, parameter.span),
                        "duplicate rule parameter '" + parameter.name + "'");
      for (const auto& type : parameter.operand_types)
        checkSignatureType(type, parameter.kind, rule.source_name);
      checkSignatureType(parameter.result_type, parameter.kind,
                         rule.source_name);
    }
  }

  void open(Scope& scope, const std::string& name, OpId id,
            const SourceOrigin& at) {
    const auto [found, inserted] = scope.operations.emplace(name, id);
    if (!inserted && found->second != id)
      context_.report(at, "ambiguous operation '" + name + "'");
  }

  void openAll(Scope& scope, const DeclarationKey& dialect,
               const SourceOrigin& at) {
    for (const auto& [name, id] : dialects_.at(dialect).operations)
      open(scope, name, id, at);
  }

  void alias(Scope& scope, Aliases& aliases, const std::string& name,
             const DeclarationKey& dialect, const SourceOrigin& at) {
    const auto [found, inserted] = aliases.emplace(name, dialect);
    if (!inserted && found->second != dialect) {
      context_.report(at, "duplicate dialect alias '" + name + "'");
      return;
    }
    for (const auto& [operation, id] : dialects_.at(dialect).operations)
      open(scope, name + "." + operation, id, at);
  }

  void openImport(Scope& scope, Aliases& aliases, const std::string& source,
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

  void openUse(Scope& scope, const Aliases& aliases, const std::string& source,
               const ast::Use& used) {
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
  TypeId tensor_;
  std::map<DeclarationKey, DialectSymbols> dialects_;
  std::map<DeclarationKey, const ast::Rule*> rules_;
  std::map<std::string, const Module*> modules_;
  std::set<std::string> active_;
  std::set<std::string> complete_;
};

}  // namespace

void resolve(AnalysisContext& context) { Resolver(context).run(); }

}  // namespace tepl::core::detail
