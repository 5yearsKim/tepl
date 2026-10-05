#pragma once

#include <filesystem>

namespace tepl::support {
// Create a minimal TEPL project, preserving existing files. Throws on failure.
void initializeProject(const std::filesystem::path& directory);
}  // namespace tepl::support
