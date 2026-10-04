#include "src/core/type/literal.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>

namespace tepl::core {
namespace {

bool fitsInteger(std::string_view spelling, bool is_signed, unsigned bits) {
  const bool negative = spelling.starts_with('-');
  if (negative && !is_signed) return false;
  if (negative || spelling.starts_with('+')) spelling.remove_prefix(1);
  std::uint64_t magnitude = 0;
  const auto parsed = std::from_chars(
      spelling.data(), spelling.data() + spelling.size(), magnitude);
  if (parsed.ec != std::errc{} ||
      parsed.ptr != spelling.data() + spelling.size())
    return false;
  const auto limit = is_signed
                         ? (std::uint64_t{1} << (bits - 1)) - (negative ? 0 : 1)
                     : bits == 64 ? std::numeric_limits<std::uint64_t>::max()
                                  : (std::uint64_t{1} << bits) - 1;
  return magnitude <= limit;
}

}  // namespace

bool validGraphLiteral(std::string_view value, DType dtype) {
  switch (dtype) {
    case DType::kBool:
      return fitsInteger(value, false, 1);
    case DType::kI8:
      return fitsInteger(value, true, 8);
    case DType::kI16:
      return fitsInteger(value, true, 16);
    case DType::kI32:
      return fitsInteger(value, true, 32);
    case DType::kI64:
      return fitsInteger(value, true, 64);
    case DType::kU8:
      return fitsInteger(value, false, 8);
    case DType::kU16:
      return fitsInteger(value, false, 16);
    case DType::kU32:
      return fitsInteger(value, false, 32);
    case DType::kU64:
      return fitsInteger(value, false, 64);
    case DType::kF16:
    case DType::kBF16:
    case DType::kF32:
    case DType::kF64:
      return true;
  }
  return false;
}

bool validHostLiteral(std::string_view value, TypeKind type) {
  if (type == TypeKind::kIndex) return fitsInteger(value, false, 64);
  if (type == TypeKind::kI64) return fitsInteger(value, true, 64);
  if (type != TypeKind::kF64) return false;
  if (value.starts_with('+')) value.remove_prefix(1);
  double number = 0;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), number);
  return parsed.ec == std::errc{} &&
         parsed.ptr == value.data() + value.size() && std::isfinite(number);
}

}  // namespace tepl::core
