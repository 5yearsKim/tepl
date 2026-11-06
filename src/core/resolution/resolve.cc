#include "src/core/resolution/rules/resolve.h"

#include "src/core/resolution/context.h"
#include "src/core/resolution/file_scope.h"
#include "src/core/resolution/resolve_context.h"
#include "src/core/resolution/rules/expand.h"
#include "src/shared/source/identity.h"
namespace tepl::core::detail {
void resolve(ResolutionContext& context) {
  DialectRegistry dialects;
  for (const auto& dialect : context.input.declarations.dialects) {
    const DeclarationKey key{sourceKey(dialect.source_name), dialect.name};
    auto [entry, added] = dialects.emplace(key, DialectSymbols{&dialect, {}});
    if (!added) {
      context.report(origin(dialect.source_name, dialect.span),
                     "duplicate dialect '" + dialect.name + "'");
      continue;
    }
    for (const auto& op : dialect.operations) {
      OpId id{context.output.operations.size()};
      context.output.operations.push_back(
          {id, dialect.source_name, dialect.name, op});
      for (const auto& name : {op.name, op.alias.value_or(op.name)}) {
        auto [it, inserted] = entry->second.operations.emplace(name, id);
        if (!inserted && it->second != id)
          context.report(origin(dialect.source_name, op.span),
                         "duplicate operation name or alias '" + name + "'");
      }
    }
  }
  const auto rules = resolveRules(context);
  buildFileScopes(context, dialects, rules);
}
}  // namespace tepl::core::detail
