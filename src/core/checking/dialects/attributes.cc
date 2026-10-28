#include "src/core/checking/dialects/attributes.h"

#include <set>

#include "src/core/checking/context.h"
#include "src/shared/source/identity.h"

namespace tepl::core::detail {

namespace {

std::vector<AttributeField> checkFields(
    CheckingContext& context, const std::vector<syntax::AttrField>& fields,
    const std::string& source) {
  std::vector<AttributeField> result;
  std::set<std::string> names;
  for (const auto& field : fields) {
    const auto at = origin(source, field.span);
    if (!names.insert(field.name).second)
      context.report(at, "duplicate attribute field '" + field.name + "'");
    static const std::set<std::string> supported = {
        "axis",           "extent", "coordinate", "dtype",     "u64",
        "string",         "i64",    "bool",       "precision", "dot_algorithm",
        "replica_groups", "region", "elements"};
    if (field.type == "index")
      context.report(at,
                     "attribute type 'index' was removed; use axis, extent, "
                     "coordinate, u64, or i64");
    else if (!supported.contains(field.type))
      context.report(at, "unknown attribute type '" + field.type + "'");
    if (field.empty_default && !field.list_depth)
      context.report(at, "empty list default requires a list type");
    auto role = field.type == "axis"         ? AttributeRole::Axis
                : field.type == "extent"     ? AttributeRole::Extent
                : field.type == "coordinate" ? AttributeRole::Coordinate
                                             : AttributeRole::General;
    auto storage = role == AttributeRole::Coordinate ? "i64"
                   : role != AttributeRole::General  ? "u64"
                                                     : field.type;
    if (role == AttributeRole::Axis && field.list_depth > 1)
      context.report(at,
                     "axis attributes must be scalar or one-dimensional lists");
    result.push_back({field.name, storage, field.list_depth, field.optional,
                      field.empty_default, at, role});
  }
  return result;
}

}  // namespace

AttributeSchemaId addAttributeSchema(
    CheckingContext& context, const syntax::Dialect& dialect,
    const std::string& name, const std::vector<syntax::AttrField>& fields,
    const SourceOrigin& at) {
  auto& schemas = context.output.attribute_schemas;
  const AttributeSchemaId id{schemas.size()};
  schemas.push_back({id, dialect.name, name,
                     checkFields(context, fields, dialect.source_name), at});
  return id;
}

SchemaNames checkAttributeSchemas(CheckingContext& context,
                                  const syntax::Dialect& dialect) {
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
