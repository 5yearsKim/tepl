#include "src/codegen/rust/names.h"

#include <set>
#include <stdexcept>

namespace tepl::codegen::rust {
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
  static const std::set<std::string> reserved = {
      "self",  "super",    "crate",  "mod",    "type",    "fn",
      "pub",   "use",      "enum",   "struct", "match",   "impl",
      "trait", "where",    "const",  "static", "move",    "ref",
      "as",    "async",    "await",  "loop",   "for",     "in",
      "let",   "mut",      "dyn",    "return", "break",   "continue",
      "if",    "else",     "true",   "false",  "extern",  "unsafe",
      "while", "abstract", "become", "box",    "do",      "final",
      "macro", "override", "priv",   "typeof", "unsized", "virtual",
      "yield", "try",      "gen"};
  if (result.empty() || result == "_" || reserved.contains(result) ||
      (result[0] >= '0' && result[0] <= '9') ||
      result.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") !=
          std::string::npos)
    throw std::invalid_argument(
        "cannot use '" + std::string(name) +
        "' as a Rust module name; rename the declaration or source file");
  return result;
}

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

std::string identifier(std::string_view name) {
  // Prefixing every host method and field avoids keywords and collisions
  // between an escaped keyword and an ordinary TEPL identifier (including
  // self/Self).
  return "tepl_" + std::string(name);
}
std::string dtype(core::DType value) {
  if (value == core::DType::kBF16) return "DType::BF16";
  return "DType::" + pascal(core::dtypeName(value));
}
std::string type(const core::Type& value, bool argument) {
  switch (value.kind) {
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
std::string capture(std::size_t id) { return "capture_" + std::to_string(id); }
std::string descriptor(std::size_t id) {
  return "descriptor_" + std::to_string(id);
}

Names::Names(const core::Program& program) {
  operations_.resize(program.operations.size());
  schemas_.resize(program.attribute_schemas.size());
  std::set<std::string> modules, variants{"None", "TeplLiteral"};
  for (const auto& declaration : program.dialects) {
    DialectNames dialect{declaration.name,
                         moduleName(declaration.name),
                         pascal(declaration.name),
                         declaration.origin.definition.source_name,
                         {},
                         {}};
    if (!modules.insert(dialect.module).second ||
        !variants.insert(dialect.variant).second)
      throw std::invalid_argument("dialect module name collision for '" +
                                  declaration.name +
                                  "'; use distinct dialect names");
    std::set<std::string> ops, schemas{"None"};
    for (const auto& op : program.operations) {
      if (op.dialect != dialect.name ||
          op.origin.definition.source_name != dialect.source)
        continue;
      auto variant = pascal(op.name);
      if (variant == "Self" || !ops.insert(variant).second)
        throw std::invalid_argument("operation variant collision in dialect '" +
                                    dialect.name + "'");
      dialect.operations.push_back(op.id);
      operations_[op.id.value] = dialect.module + "::Op::" + variant;
    }
    for (const auto& schema : program.attribute_schemas) {
      if (schema.dialect != dialect.name ||
          schema.origin.definition.source_name != dialect.source)
        continue;
      auto variant = pascal(schema.name);
      if (variant == "Self" || !schemas.insert(variant).second)
        throw std::invalid_argument("attribute variant collision in dialect '" +
                                    dialect.name + "'");
      dialect.schemas.push_back(schema.id);
      schemas_[schema.id.value] = variant;
    }
    dialects.push_back(std::move(dialect));
  }
}
std::string Names::operation(core::OpId id) const {
  return operations_.at(id.value);
}
std::string Names::schema(core::AttributeSchemaId id) const {
  return schemas_.at(id.value);
}
}  // namespace tepl::codegen::rust
