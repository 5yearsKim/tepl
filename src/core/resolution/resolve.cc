#include "src/core/resolution/resolve.h"

#include "src/core/dialect/check.h"
#include "src/core/resolution/file_scope.h"
#include "src/core/rule/resolve.h"

namespace tepl::core::detail {

void resolve(AnalysisContext& context) {
  const auto dialects = checkDialects(context);
  const auto rules = resolveRules(context);
  buildFileScopes(context, dialects, rules);
}

}  // namespace tepl::core::detail
