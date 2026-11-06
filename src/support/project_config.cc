#include "src/support/project_config.h"

#include <filesystem>
#include <initializer_list>
#include <string_view>

#include "toml.hpp"

namespace tepl::support {
ProjectConfigResult parseProjectConfig(const SourceFile& source) {
  ProjectConfigResult result;
  try {
    const auto table = toml::parse(source.text, source.identity);
    ProjectConfig config;
    config.root = std::filesystem::path(source.identity).parent_path().string();
    auto report = [&](const toml::node& node, std::string message) {
      const auto at = node.source().begin;
      result.diagnostics.emplace_back(at.line, at.column, std::move(message),
                                      source.identity);
    };
    auto knownKeys = [&](const toml::table& section, std::string_view prefix,
                         std::initializer_list<std::string_view> allowed) {
      for (const auto& [key, value] : section) {
        bool known = false;
        for (const auto candidate : allowed)
          if (key.str() == candidate) known = true;
        if (!known)
          report(value, "unknown project configuration key '" +
                            std::string(prefix) + std::string(key.str()) + "'");
      }
    };
    knownKeys(table, "", {"tepl-version", "sources", "check"});
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
    if (const auto* node = table.get("sources")) {
      if (const auto* sources = node->as_table()) {
        knownKeys(*sources, "sources.",
                  {"dialects", "rules", "graphs", "exclude"});
        auto strings = [&](const char* key, std::vector<std::string>& output) {
          const auto* value = sources->get(key);
          if (!value) return;
          const auto* array = value->as_array();
          if (!array) {
            report(*value, std::string("sources.") + key +
                               " must be an array of nonempty strings");
            return;
          }
          output.clear();
          for (const auto& item : *array) {
            const auto text = item.value<std::string>();
            if (!text || text->empty())
              report(item, std::string("sources.") + key +
                               " must contain nonempty strings");
            else
              output.push_back(*text);
          }
        };
        strings("dialects", config.sources.dialects);
        strings("rules", config.sources.rules);
        strings("graphs", config.sources.graphs);
        strings("exclude", config.sources.exclude);
      } else {
        report(*node, "sources must be a table");
      }
    }
    if (const auto* node = table.get("check")) {
      if (const auto* check = node->as_table()) {
        knownKeys(*check, "check.", {"semantic-policy"});
        if (const auto* policy = check->get("semantic-policy")) {
          const auto value = policy->value<std::string>();
          if (!value || (*value != "strict" && *value != "algebraic"))
            report(*policy,
                   "check.semantic-policy must be 'strict' or 'algebraic'");
          else
            config.semantic_policy = *value;
        }
      } else {
        report(*node, "check must be a table");
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
