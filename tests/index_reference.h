#pragma once
#include "src/core/index_semantics/ir.h"
namespace tepl::core::index_semantics {
struct Metadata {
  std::vector<List> shapes;
  std::map<std::size_t, std::vector<List>> descriptors;
};
enum class Status { Found, IllegalMetadata, Unknown };
struct Result {
  Status status = Status::Unknown;
  std::string reason;
  std::map<std::size_t, std::vector<List>> attributes;
  std::vector<std::string> trace;
};
struct Limits {
  std::size_t max_terms = 256, max_work = 100000;
};
Result infer(const Plan&, const Metadata&, Limits = {});
}  // namespace tepl::core::index_semantics
