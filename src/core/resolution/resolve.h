#pragma once

namespace tepl::core::detail {

struct AnalysisContext;

// Register checked declarations before building file visibility scopes.
void resolve(AnalysisContext& context);

}  // namespace tepl::core::detail
