#include "src/support/temporary_directory.h"

#include <random>
#include <stdexcept>
#include <string>
#include <system_error>

namespace tepl::support {
TemporaryDirectory::TemporaryDirectory(std::string_view prefix) {
  if (prefix.find_first_of("/\\") != std::string_view::npos)
    throw std::invalid_argument(
        "temporary directory prefix contains a separator");
  const auto root = std::filesystem::temp_directory_path();
  std::random_device random;
  for (int attempt = 0; attempt < 128; ++attempt) {
    const auto candidate =
        root / (std::string(prefix) + std::to_string(random()) + "-" +
                std::to_string(random()));
    std::error_code error;
    // Creation is atomic: never reuse an existing directory or symlink.
    if (std::filesystem::create_directory(candidate, error)) {
      path_ = candidate;
      return;
    }
    if (error && error != std::errc::file_exists)
      throw std::filesystem::filesystem_error("create temporary directory",
                                              candidate, error);
  }
  throw std::runtime_error("cannot create a unique temporary directory");
}

TemporaryDirectory::~TemporaryDirectory() {
  std::error_code error;
  std::filesystem::remove_all(path_, error);
}
}  // namespace tepl::support
