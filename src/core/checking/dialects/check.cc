#include "src/core/checking/dialects/check.h"

#include <set>
#include <stdexcept>
#include <utility>

#include "src/core/checking/context.h"
#include "src/core/checking/dialects/attributes.h"
#include "src/core/checking/metadata/check.h"
#include "src/core/checking/operations/semantics.h"
#include "src/shared/source/identity.h"

namespace tepl::core::detail {

namespace {
using OperationNames = std::map<std::string, OpId>;

void registerOperation(CheckingContext& context, TypeId tensor, OpId id,
                       const syntax::Dialect& dialect, const syntax::OpDecl& op,
                       const SchemaNames& schemas, OperationNames& names) {
  const auto at = origin(dialect.source_name, op.span);
  for (const auto& property : op.duplicate_properties)
    context.report(at, "duplicate operation property '" + property + "'");
  if (id.value != context.output.operations.size())
    throw std::logic_error("resolved operation IDs are not consecutive");
  Operation value;
  value.id = id;
  value.dialect = dialect.name;
  value.name = op.name;
  value.alias = op.alias;
  value.result = tensor;
  value.origin = at;
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
    if (const auto* shared = std::get_if<syntax::SharedAttrs>(&*op.attrs)) {
      const auto found = schemas.find(shared->name);
      if (found == schemas.end())
        context.report(at, "unknown attribute schema '" + shared->name + "'");
      else
        value.attributes = found->second;
    } else {
      const auto& fields = std::get<syntax::InlineAttrs>(*op.attrs).fields;
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
  if (op.axes) {
    if (!op.semantics)
      context.report(at, "axes requires semantics");
    else
      value.semantic_definition = semantics::check(
          context, value, *op.axes, *op.semantics, dialect.source_name);
  } else if (op.semantics)
    context.report(at,
                   "semantics requires an axes block; legacy element semantics "
                   "is no longer supported");
  context.output.operations.push_back(std::move(value));
  const auto addName = [&](const std::string& name) {
    if (!names.emplace(name, id).second)
      context.report(at, "duplicate operation name or alias '" + name + "'");
  };
  addName(op.name);
  if (op.alias) addName(*op.alias);
}

void registerDialect(CheckingContext& context, TypeId tensor,
                     const syntax::Dialect& dialect) {
  OperationNames names;
  context.output.dialects.push_back(
      {dialect.name, origin(dialect.source_name, dialect.span)});
  const auto schemas = checkAttributeSchemas(context, dialect);
  for (const auto& resolved : context.resolved.data().declarations)
    if (resolved.source == dialect.source_name &&
        resolved.dialect == dialect.name)
      registerOperation(context, tensor, resolved.id, dialect,
                        resolved.declaration, schemas, names);
}

}  // namespace

void checkDialects(CheckingContext& context) {
  const auto tensor = context.types.concrete({TypeKind::kTensor});
  for (const auto& dialect : context.input.declarations.dialects)
    registerDialect(context, tensor, dialect);
}

}  // namespace tepl::core::detail
