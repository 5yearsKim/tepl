#include "src/core/resolution/context.h"

#include <utility>

#include "src/shared/source/identity.h"

namespace tepl::core::detail {

void ResolutionContext::report(SourceOrigin at, std::string message,
                               std::vector<SourceLocation> related) {
  diagnostics.push_back(
      {std::move(at), std::move(message), std::move(related)});
}

std::optional<OpId> ResolutionContext::operation(const std::string& name,
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
