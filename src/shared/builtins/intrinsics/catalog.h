#pragma once

#include <span>
#include <string_view>

namespace tepl::core::intrinsics {
// Separate from public Builtin identities, even when IR spellings overlap.
enum class Intrinsic {
#define TEPL_INTRINSIC(id, ...) id,
#include "src/shared/builtins/intrinsics/catalog.def"
#undef TEPL_INTRINSIC
};
struct Signature {
  Intrinsic intrinsic;
  std::string_view name;
  std::size_t min_arity;
  int max_arity;
  bool acceptsArity(std::size_t count) const {
    return count >= min_arity &&
           (max_arity < 0 || count <= static_cast<std::size_t>(max_arity));
  }
};
std::span<const Signature> catalog();
const Signature& signature(Intrinsic);
const Signature* resolve(std::string_view);
}  // namespace tepl::core::intrinsics
