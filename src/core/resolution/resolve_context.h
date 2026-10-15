#pragma once

namespace tepl::core::detail {

struct ResolutionContext;

// Register checked declarations before building file visibility scopes.
void resolve(ResolutionContext& context);

}  // namespace tepl::core::detail
