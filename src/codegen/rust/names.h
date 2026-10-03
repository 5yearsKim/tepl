#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "src/codegen/common/project_plan.h"
#include "src/core/ir.h"

namespace tepl::codegen::rust {
std::string pascal(std::string_view name);
std::string moduleName(std::string_view name);
std::string quote(std::string_view text);
std::string identifier(std::string_view name);
std::string dtype(core::DType dtype);
std::string type(const core::Type& type, bool argument = false);
std::string capture(std::size_t id);
std::string descriptor(std::size_t id);

struct DialectNames {
  std::string name, module, variant, source;
  std::vector<core::OpId> operations;
  std::vector<core::AttributeSchemaId> schemas;
};

class Names {
 public:
  Names(const core::Program& program, const ProjectPlan& project);
  std::string operation(core::OpId id) const;
  std::string schema(core::AttributeSchemaId id) const;
  std::vector<DialectNames> dialects;

 private:
  std::vector<std::string> operations_, schemas_;
};
}  // namespace tepl::codegen::rust
