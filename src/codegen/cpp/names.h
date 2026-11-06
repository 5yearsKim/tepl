#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/codegen/common/project_plan.h"
#include "src/executable/ir.h"

namespace tepl::codegen::cpp {
std::string quote(std::string_view text);
enum class NameKind { kModule, kVariant, kField, kMethod, kRule };

// Validate target spelling before emission, retaining semantic identity.
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

struct DialectNames {
  std::string name, module, file_stem;
  std::vector<core::OpId> operations;
  std::vector<core::AttributeSchemaId> schemas;
};

struct RuleModuleNames {
  std::string path, qualified;
  std::vector<core::RuleId> rules;
};

class Names {
 public:
  Names(const executable::ProgramData& program, const ProjectPlan& project,
        std::string namespace_name);
  std::string operation(core::OpId id, std::string_view root) const;
  const std::string& operationVariant(core::OpId id) const;
  std::string schema(core::AttributeSchemaId id) const;
  const std::string& field(core::AttributeSchemaId id, std::size_t index) const;
  const std::string& rule(core::RuleId id) const;
  std::string root;
  std::vector<DialectNames> dialects;
  std::vector<RuleModuleNames> modules;
  std::map<std::string, std::vector<std::string>> module_indexes;

 private:
  std::vector<std::string> operations_, operation_variants_, schemas_, rules_;
  std::vector<std::vector<std::string>> fields_;
};
}  // namespace tepl::codegen::cpp
