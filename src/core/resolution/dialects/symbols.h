#pragma once

#include <map>
#include <string>

#include "src/core/syntax/ast.h"
#include "src/shared/catalog/ids.h"
#include "src/shared/source/identity.h"

namespace tepl::core::detail {

using OperationNames = std::map<std::string, OpId>;
using SchemaNames = std::map<std::string, AttributeSchemaId>;

struct DialectSymbols {
  const syntax::Dialect* declaration;
  OperationNames operations;
};

using DialectRegistry = std::map<DeclarationKey, DialectSymbols>;

}  // namespace tepl::core::detail
