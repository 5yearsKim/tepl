#pragma once

#include <string>
#include <vector>

#include "src/ast/ast.h"
#include "src/core/dialect/symbols.h"

namespace tepl::core::detail {

struct AnalysisContext;

// Register shared schemas and check their fields before checking operations.
SchemaNames checkAttributeSchemas(AnalysisContext& context,
                                  const ast::Dialect& dialect);
AttributeSchemaId addAttributeSchema(AnalysisContext& context,
                                     const ast::Dialect& dialect,
                                     const std::string& name,
                                     const std::vector<ast::AttrField>& fields,
                                     const SourceOrigin& at);

}  // namespace tepl::core::detail
