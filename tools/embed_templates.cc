#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
std::string read(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot read " + path);
  std::string contents{std::istreambuf_iterator<char>(input), {}};
  if (input.bad()) throw std::runtime_error("cannot read " + path);
  return contents;
}

std::string quoted(const std::string& text) {
  std::string result = "\"";
  for (const char ch : text) {
    switch (ch) {
      case '\\':
        result += "\\\\";
        break;
      case '"':
        result += "\\\"";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        result += ch;
        break;
    }
  }
  return result + '"';
}

// A template may itself contain a raw string terminator; choose an unused one.
std::string delimiter(const std::string& contents) {
  std::string result = "TEPL_TEMPLATE";
  for (unsigned index = 0;
       contents.find(")" + result + "\"") != std::string::npos; ++index) {
    result = "TEPL_" + std::to_string(index);
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || (argc - 3) % 2 != 0) {
    std::cerr << "Usage: embed_templates OUTPUT NAMESPACE [RELATIVE_PATH "
                 "INPUT_FILE]...\n";
    return 2;
  }
  try {
    std::string header =
        "#pragma once\n#include \"src/codegen/output.h\"\n"
        "namespace " +
        std::string(argv[2]) +
        " {\n"
        "inline std::vector<GeneratedFile> templateFiles() { return {\n";
    for (int index = 3; index < argc; index += 2) {
      const auto contents = read(argv[index + 1]);
      const auto marker = delimiter(contents);
      header += "{" + quoted(argv[index]) + ", R\"" + marker + "(" + contents +
                ")" + marker + "\"},\n";
    }
    header += "}; }\n}\n";
    std::ofstream output(argv[1], std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error("cannot write " + std::string(argv[1]));
    output << header;
    output.close();
    if (!output)
      throw std::runtime_error("cannot write " + std::string(argv[1]));
  } catch (const std::exception& error) {
    std::cerr << "embed_templates: " << error.what() << '\n';
    return 1;
  }
}
