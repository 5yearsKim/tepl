#pragma once

#include <map>
#include <string>
#include <vector>

#include "src/core/syntax/ast.h"
#include "src/shared/catalog/ids.h"

namespace tepl::core::detail {

struct CheckingContext;
using SchemaNames = std::map<std::string, AttributeSchemaId>;

// Register shared schemas and check their fields before checking operations.
SchemaNames checkAttributeSchemas(CheckingContext& context,
                                  const syntax::Dialect& dialect);
AttributeSchemaId addAttributeSchema(
    CheckingContext& context, const syntax::Dialect& dialect,
    const std::string& name, const std::vector<syntax::AttrField>& fields,
    const SourceOrigin& at);

}  // namespace tepl::core::detail
