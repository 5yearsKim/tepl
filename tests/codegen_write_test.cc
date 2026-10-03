#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "src/codegen/write.h"

namespace {
std::string read(const std::filesystem::path& path) {
  std::ifstream input(path);
  return {std::istreambuf_iterator<char>(input), {}};
}
void write(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path);
  output << text;
}
}  // namespace
int main() {
  namespace fs = std::filesystem;
  std::string temporary =
      (fs::temp_directory_path() / "tepl-write-test-XXXXXX").string();
  assert(mkdtemp(temporary.data()));
  const fs::path root(temporary);
  using tepl::codegen::synchronize;
  const tepl::codegen::WriteOptions check{true, false};
  const tepl::codegen::WriteOptions raw{false, false};
  std::vector<tepl::codegen::GeneratedFile> first{{"rules/old.rs", "old"},
                                                  {"mod.rs", "mod"}};
  assert(!synchronize(first, root / "missing", check).current());
  assert(!fs::exists(root / "missing"));
  synchronize(first, root, raw);
  write(root / "host.rs", "handwritten");
  assert(synchronize(first, root, check).current());
  write(root / "mod.rs", "edited");
  const auto manifest = read(root / ".tepl-generated-files");
  assert(!synchronize(first, root, check).current());
  assert(read(root / "mod.rs") == "edited");
  assert(read(root / ".tepl-generated-files") == manifest);
  std::vector<tepl::codegen::GeneratedFile> second{{"new.rs", "new"}};
  assert(!synchronize(second, root, check).current());
  assert(fs::exists(root / "rules/old.rs"));
  synchronize(second, root, raw);
  assert(!fs::exists(root / "rules"));
  assert(!fs::exists(root / "mod.rs"));
  assert(read(root / "host.rs") == "handwritten");
  assert(synchronize(second, root, check).current());
  // A corrupted manifest must not delete arbitrary source files.
  write(root / ".tepl-generated-files",
        "# TEPL generated files v1\n../outside.rs\n");
  bool rejected = false;
  try {
    synchronize(second, root, raw);
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected && read(root / "host.rs") == "handwritten");
  fs::remove(root / ".tepl-generated-files");
  fs::create_directory_symlink(root, root / "link");
  rejected = false;
  try {
    synchronize({{"link/host.rs", "overwrite"}}, root, raw);
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected && read(root / "host.rs") == "handwritten");
  fs::remove_all(root);
}
