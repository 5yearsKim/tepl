#include "src/support/project_config.h"

#include <filesystem>

#include "toml.hpp"

namespace tepl::support {
ProjectConfigResult parseProjectConfig(const SourceFile& source) {
  ProjectConfigResult result;
  try {
    const auto table = toml::parse(source.text, source.identity);
    ProjectConfig config;
    config.root = std::filesystem::path(source.identity).parent_path().string();
    if (const auto* version = table.get("tepl-version")) {
      const auto at = version->source().begin;
      config.tepl_version = version->value<std::string>();
      if (!config.tepl_version) {
        result.diagnostics.emplace_back(at.line, at.column,
                                        "tepl-version must be a string",
                                        source.identity);
      } else if (*config.tepl_version != kTeplVersion) {
        result.diagnostics.emplace_back(
            at.line, at.column,
            "unsupported tepl-version '" + *config.tepl_version +
                "'; this compiler supports '" + kTeplVersion + "'",
            source.identity);
      }
    }
    if (result.diagnostics.empty()) result.config = std::move(config);
  } catch (const toml::parse_error& error) {
    const auto at = error.source().begin;
    result.diagnostics.emplace_back(
        at.line, at.column,
        "invalid tepl.toml: " + std::string(error.description()),
        source.identity);
  }
  return result;
}
}  // namespace tepl::support
