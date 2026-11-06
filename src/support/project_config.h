#pragma once

#include <optional>
#include <string>
#include <vector>

#include "src/shared/diagnostics/diagnostic.h"
#include "src/shared/source/source.h"

namespace tepl::support {
inline constexpr char kTeplVersion[] = "0.2.0";

struct ProjectSources {
  std::vector<std::string> dialects = {"dialects"};
  std::vector<std::string> rules = {"rules"};
  std::vector<std::string> graphs = {"graphs"};
  std::vector<std::string> exclude;
};

struct ProjectConfig {
  std::string root;
  std::optional<std::string> tepl_version;
  ProjectSources sources;
  std::string semantic_policy = "strict";
};
struct ProjectConfigResult {
  std::optional<ProjectConfig> config;
  std::vector<Diagnostic> diagnostics;
};
// The directory containing tepl.toml is the project root. An empty file is
// valid.
ProjectConfigResult parseProjectConfig(const SourceFile&);
}  // namespace tepl::support
