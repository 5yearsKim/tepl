#include "src/core/dialect/attributes.h"

#include <set>

#include "src/core/analysis_context.h"
#include "src/core/resolution/source.h"

namespace tepl::core::detail {

namespace {

std::vector<AttributeField> checkFields(
    AnalysisContext& context, const std::vector<ast::AttrField>& fields,
    const std::string& source) {
  std::vector<AttributeField> result;
  std::set<std::string> names;
  for (const auto& field : fields) {
    const auto at = origin(source, field.span);
    if (!names.insert(field.name).second)
      context.report(at, "duplicate attribute field '" + field.name + "'");
    static const std::set<std::string> supported = {
        "dtype",     "index",         "string",         "i64",    "bool",
        "precision", "dot_algorithm", "replica_groups", "region", "elements"};
    if (!supported.contains(field.type))
      context.report(at, "unknown attribute type '" + field.type + "'");
    if (field.empty_default && !field.list_depth)
      context.report(at, "empty list default requires a list type");
    result.push_back({field.name, field.type, field.list_depth, field.optional,
                      field.empty_default, at});
  }
  return result;
}

}  // namespace

AttributeSchemaId addAttributeSchema(AnalysisContext& context,
                                     const ast::Dialect& dialect,
                                     const std::string& name,
                                     const std::vector<ast::AttrField>& fields,
                                     const SourceOrigin& at) {
  auto& schemas = context.output.attribute_schemas;
  const AttributeSchemaId id{schemas.size()};
  schemas.push_back({id, dialect.name, name,
                     checkFields(context, fields, dialect.source_name), at});
  return id;
}

SchemaNames checkAttributeSchemas(AnalysisContext& context,
                                  const ast::Dialect& dialect) {
  SchemaNames schemas;
  for (const auto& schema : dialect.schemas) {
    const auto at = origin(dialect.source_name, schema.span);
    const auto id =
        addAttributeSchema(context, dialect, schema.name, schema.fields, at);
    if (!schemas.emplace(schema.name, id).second)
      context.report(at, "duplicate attribute schema '" + schema.name + "'");
  }
  return schemas;
}

}  // namespace tepl::core::detail
