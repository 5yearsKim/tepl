#pragma once
#include <functional>

#include "src/shared/catalog/completed.h"
namespace tepl::core::completed {
// Canonical same-dtype/numeric contracts; do not introduce a numeric premise.
bool coreTypes(const inference::Program&,
               const std::function<const metadata::Program*(std::size_t)>&);
}  // namespace tepl::core::completed
