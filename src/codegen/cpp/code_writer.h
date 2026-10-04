#pragma once

#include <sstream>
#include <string>
#include <string_view>

namespace tepl::codegen::cpp {
class CodeWriter {
 public:
  void line(std::string_view text = {}) {
    if (!text.empty()) out_ << std::string(indent_ * 2, ' ');
    out_ << text << '\n';
  }
  void open(std::string_view text) {
    line(text.empty() ? "{" : std::string(text) + " {");
    ++indent_;
  }
  void close(std::string_view suffix = {}) {
    --indent_;
    line("}" + std::string(suffix));
  }
  std::string str() const { return out_.str(); }

 private:
  std::ostringstream out_;
  std::size_t indent_ = 0;
};
}  // namespace tepl::codegen::cpp
