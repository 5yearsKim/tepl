#pragma once
#include <functional>

#include "src/core/ir.h"
namespace tepl::codegen {
struct InferenceSyntax {
  bool rust;
  std::string root;
  std::function<std::string(core::AttributeSchemaId, const std::string&)> attrs;
  std::function<std::string(core::AttributeSchemaId, std::size_t)> field;
  std::function<std::string(core::OpId)> operation;
};
std::string emitAttributeInference(const core::Program&, const core::Rule&,
                                   const InferenceSyntax&);
}  // namespace tepl::codegen
