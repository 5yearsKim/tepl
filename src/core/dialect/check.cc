#include "src/core/dialect/check.h"

#include <set>
#include <utility>

#include "src/core/analysis_context.h"
#include "src/core/dialect/attributes.h"
#include "src/core/metadata/check.h"
#include "src/core/resolution/source.h"

namespace tepl::core::detail {

namespace {

void registerOperation(AnalysisContext& context, TypeId tensor,
                       const ast::Dialect& dialect, const ast::OpDecl& op,
                       const SchemaNames& schemas, OperationNames& names) {
  const auto at = origin(dialect.source_name, op.span);
  for (const auto& property : op.duplicate_properties)
    context.report(at, "duplicate operation property '" + property + "'");
  const OpId id{context.output.operations.size()};
  Operation value{id, dialect.name, op.name,      op.alias,
                  {}, tensor,       std::nullopt, at};
  if (op.result_type != "tensor")
    context.report(
        at, "unsupported operation result type '" + op.result_type + "'");
  std::set<std::string> operands;
  for (std::size_t i = 0; i < op.operands.size(); ++i) {
    const auto& operand = op.operands[i];
    const auto location = origin(dialect.source_name, operand.span);
    if (!operands.insert(operand.name).second)
      context.report(location, "duplicate operand '" + operand.name + "'");
    if (operand.type != "tensor")
      context.report(location, "unsupported operation operand type '" +
                                   operand.type + "'");
    if (operand.variadic && i + 1 != op.operands.size())
      context.report(location, "variadic operand must be last");
    value.operands.push_back(
        {operand.name, tensor, operand.variadic, location});
  }
  if (op.attrs) {
    if (const auto* shared = std::get_if<ast::SharedAttrs>(&*op.attrs)) {
      const auto found = schemas.find(shared->name);
      if (found == schemas.end())
        context.report(at, "unknown attribute schema '" + shared->name + "'");
      else
        value.attributes = found->second;
    } else {
      const auto& fields = std::get<ast::InlineAttrs>(*op.attrs).fields;
      if (!fields.empty())
        value.attributes = addAttributeSchema(context, dialect,
                                              op.name + "::attrs", fields, at);
    }
  }
  if (op.shape_definition)
    value.shape = metadata::check(context, value, *op.shape_definition,
                                  dialect.source_name);
  if (op.dtype_definition)
    value.dtype =
        metadata::check(context, value, *op.dtype_definition,
                        dialect.source_name, metadata::ProgramKind::DType);
  context.output.operations.push_back(std::move(value));
  const auto addName = [&](const std::string& name) {
    if (!names.emplace(name, id).second)
      context.report(at, "duplicate operation name or alias '" + name + "'");
  };
  addName(op.name);
  if (op.alias) addName(*op.alias);
}

void registerDialect(AnalysisContext& context, DialectRegistry& dialects,
                     TypeId tensor, const ast::Dialect& dialect) {
  const DeclarationKey key{sourceKey(dialect.source_name), dialect.name};
  const auto [entry, inserted] =
      dialects.emplace(key, DialectSymbols{&dialect, {}});
  if (!inserted) {
    context.report(origin(dialect.source_name, dialect.span),
                   "duplicate dialect '" + dialect.name + "'");
    return;
  }
  context.output.dialects.push_back(
      {dialect.name, origin(dialect.source_name, dialect.span)});
  const auto schemas = checkAttributeSchemas(context, dialect);
  for (const auto& op : dialect.operations)
    registerOperation(context, tensor, dialect, op, schemas,
                      entry->second.operations);
}

}  // namespace

DialectRegistry checkDialects(AnalysisContext& context) {
  DialectRegistry dialects;
  const auto tensor = context.types.concrete({TypeKind::kTensor});
  for (const auto& dialect : context.input.dialects)
    registerDialect(context, dialects, tensor, dialect);
  return dialects;
}

}  // namespace tepl::core::detail
