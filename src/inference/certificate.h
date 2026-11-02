#pragma once
#include <array>

#include "src/inference/emission.h"
#include "src/inference/extensions.h"
namespace tepl::inference {
struct LayoutFacts {
  std::map<std::size_t, std::size_t> ranks;
  std::map<std::pair<std::size_t, std::size_t>, List> fields;
  std::set<std::array<std::size_t, 4>> batch_prefixes;
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> lengths;
  // {capture, dot descriptor, batch field}: rank(capture) = len(batch)+2.
  std::set<std::array<std::size_t, 3>> batch_plus_two;
  std::vector<MetadataEquality> equalities;
  bool inconsistent = false;
};
// Sound sufficient guard checker. Exact axis-layout abstraction, not numeric
// tensor testing: all equivalence classes/orders depend only on ranks/fields.
bool certify(const Program&, const LayoutFacts&, const Limits& = {});
bool sumMovement(const Program&);
}  // namespace tepl::inference
