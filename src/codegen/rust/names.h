#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/codegen/common/project_plan.h"
#include "src/core/ir.h"

namespace tepl::codegen::rust {
std::string quote(std::string_view text);
enum class NameKind { kModule, kVariant, kField, kMethod, kRule };

// Target spelling and semantic identity are separate: r#type names type.rs.
struct Identifier {
  std::string key;
  std::string token;
  static Identifier make(std::string_view source, NameKind kind,
                         const SourceOrigin& origin);
};

class NameError : public std::invalid_argument {
 public:
  NameError(SourceOrigin origin, std::string message)
      : std::invalid_argument(std::move(message)), origin(std::move(origin)) {}
  SourceOrigin origin;
};
std::string dtype(core::DType dtype);
std::string type(const core::Type& type, bool argument = false);
std::string capture(std::size_t id);
std::string descriptor(std::size_t id);

struct DialectNames {
  std::string name, module, file_stem, variant, source;
  std::vector<core::OpId> operations;
  std::vector<core::AttributeSchemaId> schemas;
};

struct RuleModuleNames {
  std::string path, qualified, source, root;
  std::vector<core::RuleId> rules;
};

// Construct the complete naming plan before any emitter runs.
class Names {
 public:
  Names(const core::Program& program, const ProjectPlan& project);
  std::string operation(core::OpId id, std::string_view root) const;
  const std::string& operationVariant(core::OpId id) const;
  std::string schema(core::AttributeSchemaId id) const;
  const std::string& field(core::AttributeSchemaId id, std::size_t index) const;
  const std::string& host(core::HostFunctionId id) const;
  const std::string& rule(core::RuleId id) const;
  std::vector<DialectNames> dialects;
  std::vector<RuleModuleNames> modules;
  std::map<std::string, std::vector<std::string>> module_indexes;

 private:
  std::vector<std::string> operations_, operation_variants_, schemas_, hosts_,
      rules_;
  std::vector<std::vector<std::string>> fields_;
};
}  // namespace tepl::codegen::rust
