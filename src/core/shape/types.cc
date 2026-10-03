#include "src/core/shape/types.h"

#include <string_view>

namespace tepl::core::shape {

std::string typeName(Type type) {
  std::string name = type.kind == Type::Kind::kInteger ? "Integer" : "Bool";
  for (std::size_t i = 0; i < type.list_depth; ++i) name = "List<" + name + ">";
  return name;
}

bool validIntegerLiteral(const std::string& spelling) {
  std::string_view digits = spelling;
  const bool negative = digits.starts_with('-');
  if (negative || digits.starts_with('+')) digits.remove_prefix(1);
  if (digits.empty() ||
      digits.find_first_not_of("0123456789") != std::string_view::npos)
    return false;
  const auto first = digits.find_first_not_of('0');
  digits = first == std::string_view::npos ? std::string_view("0")
                                           : digits.substr(first);
  const std::string_view limit =
      negative ? "170141183460469231731687303715884105728"
               : "170141183460469231731687303715884105727";
  return digits.size() < limit.size() ||
         (digits.size() == limit.size() && digits <= limit);
}

}  // namespace tepl::core::shape
