#pragma once

#include "src/inference/emission.h"
#include "src/inference/extensions.h"
namespace tepl::inference {
struct MetadataFacts {
  std::map<std::size_t, std::size_t> ranks;
  std::map<std::pair<std::size_t, std::size_t>, List> fields;
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> lengths;
  std::vector<MetadataEquality> equalities;
};
}  // namespace tepl::inference
