#include "src/codegen/write.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace tepl::codegen {
namespace {
namespace fs = std::filesystem;
constexpr auto kManifest = ".tepl-generated-files";
constexpr auto kHeader = "# TEPL generated files v1";

std::string read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot read " + path.string());
  std::string result{std::istreambuf_iterator<char>(input), {}};
  if (input.bad()) throw std::runtime_error("cannot read " + path.string());
  return result;
}
void write(const fs::path& path, const std::string& contents) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output << contents;
  output.close();
  if (!output) throw std::runtime_error("cannot write " + path.string());
}
void validate(const std::string& name, const fs::path& root) {
  const fs::path path(name);
  if (name.empty() || name == kManifest || path.is_absolute() ||
      path.generic_string() != name ||
      name.find_first_of("\r\n\\") != std::string::npos)
    throw std::runtime_error("invalid generated path: " + name);
  auto current = root;
  for (const auto& part : path) {
    if (part == ".." || part == "." || part.empty())
      throw std::runtime_error("invalid generated path: " + name);
    current /= part;
    if (fs::is_symlink(current))
      throw std::runtime_error("generated path crosses a symlink: " + name);
  }
}
struct TemporaryDirectory {
  fs::path path;
  TemporaryDirectory() {
    std::string name =
        (fs::temp_directory_path() / "tepl-format-XXXXXX").string();
    if (!mkdtemp(name.data()))
      throw std::system_error(errno, std::generic_category(), "mkdtemp");
    path = name;
  }
  ~TemporaryDirectory() {
    std::error_code error;
    fs::remove_all(path, error);
  }
};
void format(std::map<std::string, std::string>& files, Target target) {
  if (target != Target::kRust)
    throw std::runtime_error("formatting is only implemented for Rust");
  TemporaryDirectory stage;
  std::vector<std::string> arguments{"rustfmt", "--edition", "2024", "--config",
                                     "skip_children=true"};
  for (const auto& [name, contents] : files) {
    if (fs::path(name).extension() != ".rs") continue;
    auto path = stage.path / name;
    write(path, contents);
    arguments.push_back(path.string());
  }
  if (arguments.size() == 5) return;
  std::vector<char*> argv;
  for (auto& argument : arguments) argv.push_back(argument.data());
  argv.push_back(nullptr);
  const auto child = fork();
  if (child < 0)
    throw std::system_error(errno, std::generic_category(), "fork");
  if (child == 0) {
    execvp(argv.front(), argv.data());
    _exit(127);
  }
  int status;
  while (waitpid(child, &status, 0) < 0)
    if (errno != EINTR)
      throw std::system_error(errno, std::generic_category(), "waitpid");
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error(
        "rustfmt failed; install rustfmt or use --no-format");
  for (auto& [name, contents] : files)
    if (fs::path(name).extension() == ".rs") contents = read(stage.path / name);
}
}  // namespace

WriteResult synchronize(const std::vector<GeneratedFile>& input,
                        const fs::path& directory,
                        const WriteOptions& options) {
  const auto root = fs::absolute(directory).lexically_normal();
  std::map<std::string, std::string> files;
  for (const auto& file : input) {
    validate(file.path, root);
    if (!files.emplace(file.path, file.contents).second)
      throw std::runtime_error("duplicate generated path: " + file.path);
  }
  if (fs::is_symlink(root / kManifest))
    throw std::runtime_error("generated manifest is a symlink");
  std::set<std::string> previous;
  if (fs::exists(root / kManifest)) {
    std::ifstream manifest(root / kManifest);
    std::string line;
    if (!std::getline(manifest, line) || line != kHeader)
      throw std::runtime_error("invalid generated manifest");
    while (std::getline(manifest, line)) {
      validate(line, root);
      if (!previous.insert(line).second)
        throw std::runtime_error("duplicate manifest path");
    }
    if (manifest.bad())
      throw std::runtime_error("cannot read generated manifest");
  }
  if (options.format) format(files, options.target);
  std::string manifest = std::string(kHeader) + '\n';
  WriteResult result;
  for (const auto& [name, contents] : files) {
    manifest += name + '\n';
    if (!fs::exists(root / name) || read(root / name) != contents)
      result.differences.push_back(name);
  }
  for (const auto& name : previous)
    if (!files.contains(name))
      result.differences.push_back(name + " (obsolete)");
  if (!fs::exists(root / kManifest) || read(root / kManifest) != manifest)
    result.differences.push_back(kManifest);
  if (options.check) return result;
  for (const auto& [name, contents] : files) write(root / name, contents);
  for (const auto& name : previous) {
    if (files.contains(name)) continue;
    fs::remove(root / name);
    auto parent = (root / name).parent_path();
    while (parent != root && fs::is_directory(parent) && fs::is_empty(parent)) {
      fs::remove(parent);
      parent = parent.parent_path();
    }
  }
  write(root / kManifest, manifest);
  return result;
}
}  // namespace tepl::codegen
