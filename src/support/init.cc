#include "src/support/init.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>

namespace tepl::support {
namespace {
struct StarterFile {
  std::filesystem::path path;
  const char* source;
};

bool existsWithoutFollowingSymlinks(const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error == std::errc::no_such_file_or_directory) return false;
  if (error)
    throw std::filesystem::filesystem_error("cannot inspect path", path, error);
  return std::filesystem::exists(status);
}

void writeNewFile(const StarterFile& file) {
  // Exclusive creation also protects files created after the preflight checks.
  auto* output = std::fopen(file.path.string().c_str(), "wbx");
  if (!output)
    throw std::runtime_error(file.path.string() +
                             ": cannot create file: " + std::strerror(errno));
  const bool written = std::fputs(file.source, output) >= 0;
  const bool closed = std::fclose(output) == 0;
  if (!written || !closed) {
    std::error_code ignored;
    std::filesystem::remove(file.path, ignored);
    throw std::runtime_error(file.path.string() + ": cannot write file");
  }
}
}  // namespace

void initializeProject(const std::filesystem::path& directory) {
  namespace fs = std::filesystem;
  if (directory.empty())
    throw std::invalid_argument("project directory must not be empty");
  const std::array directories{directory, directory / "dialects",
                               directory / "rules"};
  const std::array files{
      StarterFile{directory / "dialects" / "your_dialect.tepl",
                  R"(dialect YourDialect {
    op add(lhs: tensor, rhs: tensor) -> tensor;
}
)"},
      StarterFile{
          directory / "rules" / "your_rule.tepl",
          R"(from "../dialects/your_dialect.tepl" import YourDialect as d;

rule commute_add {
    (d.add X Y) => (d.add Y X)
}
)"}};

  for (const auto& path : directories)
    if (existsWithoutFollowingSymlinks(path) && !fs::is_directory(path))
      throw std::runtime_error(path.string() + ": not a directory");
  for (const auto& file : files)
    if (existsWithoutFollowingSymlinks(file.path))
      throw std::runtime_error(file.path.string() +
                               ": already exists; refusing to overwrite");

  for (const auto& path : directories) fs::create_directories(path);
  std::size_t created = 0;
  try {
    for (const auto& file : files) {
      writeNewFile(file);
      ++created;
    }
  } catch (...) {
    for (std::size_t index = 0; index < created; ++index) {
      std::error_code ignored;
      fs::remove(files[index].path, ignored);
    }
    throw;
  }
}
}  // namespace tepl::support
