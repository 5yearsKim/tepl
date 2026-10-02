#pragma once

#include <string>
#include <utility>

#include "src/source.h"

namespace tepl::core::detail {

// File identity and declared name are separate keys; no delimiter encoding.
using DeclarationKey = std::pair<std::string, std::string>;

std::string sourceKey(const std::string& name);
std::string importTarget(const std::string& source, const std::string& path);
SourceOrigin origin(const std::string& source, SourceSpan span);

}  // namespace tepl::core::detail
