#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/support/process.h"
#include "src/support/temporary_directory.h"

namespace {
std::string utf8(const std::filesystem::path& path) {
  const auto text = path.u8string();
  return {text.begin(), text.end()};
}

std::string encoded(const std::string& value) {
  // ASCII encoding avoids Windows text streams translating embedded newlines.
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (const unsigned char byte : value) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result + ':';
}

int child(int argc, char** argv) {
  const std::string mode = argv[1];
  if (mode == "--arguments") {
    for (int index = 2; index < argc; ++index)
      std::cout << encoded(argv[index]);
    return 0;
  }
  if (mode == "--output") {
    // Exceeds ordinary pipe capacity on both stdout and stderr.
    for (int index = 0; index < 128; ++index) {
      std::cout << std::string(4096, 'o');
      std::cerr << std::string(4096, 'e');
    }
    return 37;
  }
  if (mode == "--stdin") {
    const auto environment = std::getenv("TEPL_TEST_INHERIT");
    assert(environment && std::string(environment) == "present");
    assert(std::cin.peek() == std::char_traits<char>::eof());
    std::cout << "stdin closed; environment inherited";
    return 0;
  }
  return 1;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) return child(argc, argv);
  namespace fs = std::filesystem;
  using tepl::support::runProcess;
  using tepl::support::TemporaryDirectory;

  fs::path removed;
  {
    TemporaryDirectory first("tepl-support-test-");
    TemporaryDirectory second("tepl-support-test-");
    assert(first.path() != second.path());
    assert(fs::is_directory(first.path()) && fs::is_directory(second.path()));
    removed = second.path();
    fs::create_directories(removed / "nested");
    std::ofstream(removed / "nested/file") << "cleanup";

    const auto executable =
        first.path() / fs::path(u8"helper's space \u03bb.exe");
    fs::copy_file(fs::absolute(argv[0]), executable);
    const std::vector<std::string> values{
        "",     "space in argument", "a\"b",  "trailing\\",
        "\\\"", "$(echo bad)",       "$HOME", "line\nbreak"};
    std::vector<std::string> arguments{utf8(executable), "--arguments"};
    arguments.insert(arguments.end(), values.begin(), values.end());
    const auto echoed = runProcess(arguments);
    std::string expected;
    for (const auto& value : values) expected += encoded(value);
    if (echoed.exit_code != 0 || echoed.output != expected)
      std::cerr << "Echo returned " << echoed.exit_code << ": "
                << echoed.output.size() << " bytes, expected "
                << expected.size() << "; output: " << echoed.output << '\n';
    assert(echoed.exit_code == 0 && echoed.output == expected);

    const auto output = runProcess({utf8(executable), "--output"});
    assert(output.exit_code == 37);
    assert(output.output.size() == 128 * 4096 * 2);
    assert(std::count(output.output.begin(), output.output.end(), 'o') ==
           128 * 4096);
    assert(std::count(output.output.begin(), output.output.end(), 'e') ==
           128 * 4096);
    const auto input = runProcess({utf8(executable), "--stdin"});
    assert(input.exit_code == 0 &&
           input.output == "stdin closed; environment inherited");

    bool rejected = false;
    try {
      runProcess({utf8(first.path() / "missing-executable")});
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    assert(rejected);
    for (const auto& invalid : std::vector<std::vector<std::string>>{
             {}, {""}, {"a", std::string("b\0c", 3)}}) {
      rejected = false;
      try {
        runProcess(invalid);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      assert(rejected);
    }
  }
  assert(!fs::exists(removed));
}
