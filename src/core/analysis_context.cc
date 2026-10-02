#include "src/core/analysis_context.h"

#include <utility>

#include "src/core/resolution/source.h"

namespace tepl::core::detail {

void AnalysisContext::report(SourceOrigin at, std::string message,
                             std::vector<SourceLocation> related) {
  diagnostics.push_back(
      {std::move(at), std::move(message), std::move(related)});
}

HostFunctionId AnalysisContext::host(const std::string& name, std::size_t arity,
                                     const SourceOrigin& at) {
  const auto key = sourceKey(at.definition.source_name) + "\n" + name;
  if (const auto found = host_names.find(key); found != host_names.end()) {
    const auto& fn = output.host_functions[found->second.value];
    if (fn.signature.arguments.size() != arity)
      report(at, "conflicting arity for host function '" + name + "'",
             {fn.origin.definition});
    return found->second;
  }
  const HostFunctionId id{output.host_functions.size()};
  Signature signature;
  for (std::size_t i = 0; i < arity; ++i)
    signature.arguments.push_back(types.variable(at));
  signature.result = types.variable(at);
  output.host_functions.push_back({id, name, std::move(signature), true, at});
  host_names.emplace(key, id);
  return id;
}

std::optional<OpId> AnalysisContext::operation(const std::string& name,
                                               const std::string& source,
                                               const SourceOrigin& at) {
  const auto scope = scopes.find(sourceKey(source));
  if (scope == scopes.end()) {
    report(at, "no scope loaded for source '" + source + "'");
    return std::nullopt;
  }
  const auto& names = scope->second.operations;
  const auto found = names.find(name);
  if (found == names.end()) {
    report(at, "unknown operation '" + name +
                   "'; semantic analysis requires a "
                   "visible dialect declaration");
    return std::nullopt;
  }
  return found->second;
}

}  // namespace tepl::core::detail
