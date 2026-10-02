#pragma once

#include <map>
#include <string>

#include "src/ast/ast.h"
#include "src/core/ids.h"
#include "src/core/resolution/source.h"

namespace tepl::core::detail {

using OperationNames = std::map<std::string, OpId>;
using SchemaNames = std::map<std::string, AttributeSchemaId>;

struct DialectSymbols {
  const ast::Dialect* declaration;
  OperationNames operations;
};

using DialectRegistry = std::map<DeclarationKey, DialectSymbols>;

}  // namespace tepl::core::detail
