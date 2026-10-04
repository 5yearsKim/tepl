#include "src/codegen/write.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include "src/support/process.h"
#include "src/support/temporary_directory.h"

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
void format(std::map<std::string, std::string>& files, Target target) {
  if (target != Target::kRust && target != Target::kCpp)
    throw std::runtime_error("formatting is not implemented for this target");
  support::TemporaryDirectory stage("tepl-format-");
  const bool cpp = target == Target::kCpp;
  const std::string formatter = cpp ? "clang-format" : "rustfmt";
  std::vector<std::string> arguments =
      cpp ? std::vector<std::string>{formatter, "--style=Google", "-i"}
          : std::vector<std::string>{formatter, "--edition", "2024", "--config",
                                     "skip_children=true"};
  const auto argument_count = arguments.size();
  const auto extension = cpp ? ".h" : ".rs";
  for (const auto& [name, contents] : files) {
    if (fs::path(name).extension() != extension) continue;
    auto path = stage.path() / name;
    write(path, contents);
    const auto utf8 = path.u8string();
    arguments.emplace_back(utf8.begin(), utf8.end());
  }
  if (arguments.size() == argument_count) return;
  try {
    const auto result = support::runProcess(arguments);
    std::cerr << result.output;
    if (result.exit_code != 0)
      throw std::runtime_error("formatter exited with code " +
                               std::to_string(result.exit_code));
  } catch (const std::exception& error) {
    throw std::runtime_error(
        formatter + " failed; install " + formatter +
        " or use --no-format: " + std::string(error.what()));
  }
  for (auto& [name, contents] : files)
    if (fs::path(name).extension() == extension)
      contents = read(stage.path() / name);
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
