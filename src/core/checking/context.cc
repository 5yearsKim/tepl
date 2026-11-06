#include "src/core/checking/context.h"

#include <utility>

namespace tepl::core::detail {

void CheckingContext::report(SourceOrigin at, std::string message,
                             std::vector<SourceLocation> related) {
  diagnostics.push_back(
      {std::move(at), std::move(message), std::move(related)});
}

}  // namespace tepl::core::detail
