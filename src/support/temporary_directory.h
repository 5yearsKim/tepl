#pragma once

#include <filesystem>
#include <string_view>

namespace tepl::support {
// Owns a uniquely created directory and removes it on scope exit.
class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view prefix = "tepl-");
  ~TemporaryDirectory();
  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};
}  // namespace tepl::support
