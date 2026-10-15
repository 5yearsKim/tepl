#include "src/core/intrinsics/catalog.h"

#include <stdexcept>

namespace tepl::core::intrinsics {
namespace {
constexpr Signature kIntrinsics[] = {
#define TEPL_INTRINSIC(id, name, min, max) {Intrinsic::id, name, min, max},
#include "src/core/intrinsics/catalog.def"
#undef TEPL_INTRINSIC
};
}  // namespace
std::span<const Signature> catalog() { return kIntrinsics; }
const Signature& signature(Intrinsic id) {
  for (const auto& entry : kIntrinsics)
    if (entry.intrinsic == id) return entry;
  throw std::logic_error("invalid intrinsic ID");
}
const Signature* resolve(std::string_view name) {
  for (const auto& entry : kIntrinsics)
    if (entry.name == name) return &entry;
  return nullptr;
}
}  // namespace tepl::core::intrinsics
