#include "src/core/resolution/source.h"

#include <filesystem>

namespace tepl::core::detail {

std::string sourceKey(const std::string& name) {
  return std::filesystem::absolute(name).lexically_normal().string();
}

std::string importTarget(const std::string& source, const std::string& path) {
  return sourceKey(
      (std::filesystem::path(source).parent_path() / path).string());
}

SourceOrigin origin(const std::string& source, SourceSpan span) {
  return {{source, span}, {}};
}

}  // namespace tepl::core::detail
