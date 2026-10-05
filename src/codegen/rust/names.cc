#include "src/codegen/rust/names.h"

#include <map>
#include <set>
#include <stdexcept>

#include "src/codegen/common/rule_plan.h"
#include "src/codegen/rust/paths.h"

namespace tepl::codegen::rust {
namespace {
const std::set<std::string_view>& rustKeywords() {
  static const std::set<std::string_view> keywords = {
      "self",     "Self",  "super",    "crate",  "mod",    "type",
      "fn",       "pub",   "use",      "enum",   "struct", "match",
      "impl",     "trait", "where",    "const",  "static", "move",
      "ref",      "as",    "async",    "await",  "loop",   "for",
      "in",       "let",   "mut",      "dyn",    "return", "break",
      "continue", "if",    "else",     "true",   "false",  "extern",
      "unsafe",   "while", "abstract", "become", "box",    "do",
      "final",    "macro", "override", "priv",   "typeof", "unsized",
      "virtual",  "yield", "try",      "gen"};
  return keywords;
}
std::string pascal(std::string_view name) {
  std::string result;
  bool upper = true;
  for (char c : name) {
    if (c == '_' || c == ':' || c == '.') {
      upper = true;
      continue;
    }
    result += upper && c >= 'a' && c <= 'z' ? char(c - 'a' + 'A') : c;
    upper = false;
  }
  if (result.empty() || (result[0] >= '0' && result[0] <= '9'))
    result = "Tepl" + result;
  return result;
}
std::string moduleName(std::string_view name) {
  std::string result;
  for (std::size_t i = 0; i < name.size(); ++i) {
    char c = name[i];
    if (c >= 'A' && c <= 'Z') {
      if (i && name[i - 1] != '_' &&
          ((name[i - 1] >= 'a' && name[i - 1] <= 'z') ||
           (i + 1 < name.size() && name[i + 1] >= 'a' && name[i + 1] <= 'z')))
        result += '_';
      c = char(c - 'A' + 'a');
    }
    result += c;
  }
  if (result.empty() || result == "_" ||
      (result[0] >= '0' && result[0] <= '9') ||
      result.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") !=
          std::string::npos)
    throw std::invalid_argument(
        "cannot use '" + std::string(name) +
        "' as a Rust module name; rename the declaration or source file");
  return result;
}

}  // namespace

std::string quote(std::string_view text) {
  std::string result = "\"";
  for (unsigned char c : text) {
    switch (c) {
      case '\\':
        result += "\\\\";
        break;
      case '"':
        result += "\\\"";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (c < 32 || c == 127) {
          const char* hex = "0123456789abcdef";
          result += "\\u{";
          result += hex[c >> 4];
          result += hex[c & 15];
          result += '}';
        } else
          result += char(c);
    }
  }
  return result + '"';
}

Identifier Identifier::make(std::string_view source, NameKind kind,
                            const SourceOrigin& origin) {
  std::string key;
  try {
    key = kind == NameKind::kModule    ? moduleName(source)
          : kind == NameKind::kVariant ? pascal(source)
          : kind == NameKind::kRule    ? "rule_" + std::string(source)
                                       : std::string(source);
  } catch (const std::invalid_argument& error) {
    throw NameError(origin, error.what());
  }
  if (key == "self" || key == "Self" || key == "super" || key == "crate" ||
      key == "_")
    throw NameError(origin, "cannot generate Rust identifier '" + key +
                                "' from TEPL name '" + std::string(source) +
                                "'; Rust does not allow r#" + key +
                                "; rename it in TEPL");
  if (key.empty() || (key[0] >= '0' && key[0] <= '9') ||
      key.find_first_not_of(
          "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") !=
          std::string::npos)
    throw NameError(origin, "invalid Rust identifier from TEPL name '" +
                                std::string(source) + "'");
  return {key, (rustKeywords().contains(key) ? "r#" : "") + key};
}
std::string dtype(core::DType value) {
  if (value == core::DType::kBF16) return "DType::BF16";
  return "DType::" + pascal(core::dtypeName(value));
}
std::string type(const core::Type& value, bool argument) {
  switch (value.kind) {
    case core::TypeKind::kDType:
      return "DType";
    case core::TypeKind::kTensor:
      return argument ? "&TensorInfo" : "TensorInfo";
    case core::TypeKind::kIndex:
      return "u64";
    case core::TypeKind::kIndexList:
      return argument ? "&[u64]" : "Vec<u64>";
    case core::TypeKind::kBool:
      return "bool";
    case core::TypeKind::kI64:
      return "i64";
    case core::TypeKind::kF64:
      return "f64";
    case core::TypeKind::kDescriptor:
      return argument ? "&OpAttrs" : "OpAttrs";
  }
  return "()";
}
std::string descriptor(std::size_t id) {
  return "descriptor_" + std::to_string(id);
}

namespace {
// Each instance represents one Rust declaration scope. Raw spelling never
// changes identity; diagnostics name both declarations involved in a collision.
class Scope {
 public:
  void insertOnce(const Identifier& id, const std::string& source,
                  const SourceOrigin& origin, std::string_view scope) {
    if (auto entry = names_.find(id.key);
        entry != names_.end() && entry->second.source == source)
      return;
    insert(id, source, origin, scope);
  }
  void reserve(std::string key) {
    names_.emplace(std::move(key), Entry{"generated runtime", {}});
  }
  void insert(const Identifier& id, const std::string& source,
              const SourceOrigin& origin, std::string_view scope) {
    const auto [entry, inserted] =
        names_.emplace(id.key, Entry{source, origin});
    if (!inserted) {
      const auto& previous = entry->second.origin.definition;
      const auto location =
          previous.source_name.empty()
              ? ""
              : " (first declared at " + previous.source_name + ":" +
                    std::to_string(previous.span.begin.line) + ":" +
                    std::to_string(previous.span.begin.column) + ")";
      throw NameError(origin, std::string(scope) + " name collision: '" +
                                  source + "' and '" + entry->second.source +
                                  "' both generate '" + id.key + "'" +
                                  location);
    }
  }

 private:
  struct Entry {
    std::string source;
    SourceOrigin origin;
  };
  std::map<std::string, Entry> names_;
};
}  // namespace

Names::Names(const core::Program& program, const ProjectPlan& project) {
  operations_.resize(program.operations.size());
  operation_variants_.resize(program.operations.size());
  schemas_.resize(program.attribute_schemas.size());
  fields_.resize(program.attribute_schemas.size());
  hosts_.resize(program.host_functions.size());
  rules_.resize(program.rules.size());
  Scope module_scope, variant_scope;
  module_scope.reserve("mod");
  for (const auto* reserved : {"None", "Literal", "Input"})
    variant_scope.reserve(reserved);
  for (std::size_t i = 0; i < project.dialects.size(); ++i) {
    const auto& declaration = project.dialects[i];
    const auto& origin = program.dialects.at(i).origin;
    const auto module =
        Identifier::make(declaration.name, NameKind::kModule, origin);
    const auto variant =
        Identifier::make(declaration.name, NameKind::kVariant, origin);
    module_scope.insert(module, declaration.name, origin, "dialect module");
    variant_scope.insert(variant, declaration.name, origin, "dialect variant");
    DialectNames dialect{declaration.name, module.token, module.key,
                         variant.token,    {},           {}};
    Scope ops, schemas;
    schemas.reserve("None");
    for (auto id : declaration.operations) {
      const auto& op = program.operations.at(id.value);
      const auto name =
          Identifier::make(op.name, NameKind::kVariant, op.origin);
      ops.insert(name, op.name, op.origin,
                 "operation variant in dialect '" + dialect.name + "'");
      dialect.operations.push_back(op.id);
      operations_[op.id.value] = dialect.module + "::Op::" + name.token;
      operation_variants_[op.id.value] = name.token;
    }
    for (auto id : declaration.schemas) {
      const auto& schema = program.attribute_schemas.at(id.value);
      const auto name =
          Identifier::make(schema.name, NameKind::kVariant, schema.origin);
      schemas.insert(name, schema.name, schema.origin,
                     "attribute variant in dialect '" + dialect.name + "'");
      dialect.schemas.push_back(schema.id);
      schemas_[schema.id.value] = name.token;
      Scope fields;
      for (const auto& field : schema.fields) {
        const auto field_name =
            Identifier::make(field.name, NameKind::kField, field.origin);
        fields.insert(field_name, field.name, field.origin, "attribute field");
        fields_[id.value].push_back(field_name.token);
      }
    }
    dialects.push_back(std::move(dialect));
  }
  for (const auto& host : program.host_functions)
    hosts_[host.id.value] =
        Identifier::make(host.name, NameKind::kMethod, host.origin).token;

  // Validate both source module scopes and filesystem scopes, including parent
  // directories. Different source spellings must never silently merge.
  std::map<std::string, Scope> child_scopes;
  std::map<std::string, SourceOrigin> files, directories;
  module_indexes[""];
  std::map<std::string, std::set<std::string>> index_children;
  for (const auto& source : project.modules) {
    const auto& origin = program.rules.at(source.rules.front().value).origin;
    RuleModuleNames module{{}, {}, "super::super", source.rules};
    std::string raw_parent;
    for (std::size_t i = 0; i < source.path.size(); ++i) {
      const auto& part = source.path[i];
      const auto name = Identifier::make(part, NameKind::kModule, origin);
      auto& scope = child_scopes[module.path];
      // A shared directory is visited for every file below it. Insert each
      // original path only once, but reject another spelling with the same key.
      const auto raw_path = raw_parent + "/" + part;
      if (name.key == "mod")
        throw NameError(origin, "rule module name collision: '" + part +
                                    "' uses reserved output file mod.rs");
      // Scope::insertOnce permits the same source directory, not duplicates
      // produced by normalization of two distinct source paths.
      scope.insertOnce(name, raw_path, origin, "rule module");
      raw_parent = raw_path;
      index_children[module.path].insert(name.token);
      if (!module.path.empty()) {
        module.path += '/';
        module.qualified += "::";
        module.root = paths::parent(module.root);
      }
      module.path += name.key;
      module.qualified += name.key;
      if (i + 1 < source.path.size()) directories.emplace(module.path, origin);
    }
    if (!files.emplace(module.path, origin).second)
      throw NameError(origin,
                      "rule module name collision for '" + module.path + "'");
    Scope rule_scope;
    for (const auto id : source.rules) {
      const auto& rule = program.rules.at(id.value);
      const auto name =
          Identifier::make(rule.name, NameKind::kRule, rule.origin);
      rule_scope.insert(name, rule.name, rule.origin, "rule");
      rules_[id.value] = name.token;
      Scope methods;
      for (const auto host_id : planRule(rule).host_functions) {
        const auto& host = program.host_functions.at(host_id.value);
        const auto name =
            Identifier::make(host.name, NameKind::kMethod, host.origin);
        methods.insert(name, host.name, host.origin,
                       "host function within rule '" + rule.name + "'");
      }
    }
    modules.push_back(std::move(module));
  }
  for (const auto& [path, children] : index_children)
    module_indexes[path] = {children.begin(), children.end()};
  for (const auto& [path, origin] : files)
    if (directories.contains(path))
      throw NameError(
          origin, "rule file/directory module collision for '" + path + "'");
}
std::string Names::operation(core::OpId id, std::string_view root) const {
  return paths::within(root, "dialects::" + operations_.at(id.value));
}
const std::string& Names::operationVariant(core::OpId id) const {
  return operation_variants_.at(id.value);
}
std::string Names::schema(core::AttributeSchemaId id) const {
  return schemas_.at(id.value);
}
const std::string& Names::field(core::AttributeSchemaId id,
                                std::size_t index) const {
  return fields_.at(id.value).at(index);
}
const std::string& Names::host(core::HostFunctionId id) const {
  return hosts_.at(id.value);
}
const std::string& Names::rule(core::RuleId id) const {
  return rules_.at(id.value);
}
}  // namespace tepl::codegen::rust
