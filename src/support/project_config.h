#pragma once

#include <optional>
#include <string>
#include <vector>

#include "src/shared/diagnostics/diagnostic.h"
#include "src/shared/source/source.h"

namespace tepl::support {
inline constexpr char kTeplVersion[] = "0.1.0";

struct ProjectConfig {
  std::string root;
  std::optional<std::string> tepl_version;
};
struct ProjectConfigResult {
  std::optional<ProjectConfig> config;
  std::vector<Diagnostic> diagnostics;
};
// The directory containing tepl.toml is the project root. An empty file is
// valid.
ProjectConfigResult parseProjectConfig(const SourceFile&);
}  // namespace tepl::support
