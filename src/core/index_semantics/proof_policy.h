#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace tepl::core::index_semantics {

// These bounds limit incomplete proof attempts, not the meaning of a rule.
struct ProofLimits {
  static constexpr std::size_t expression_depth = 256;
  static constexpr std::size_t evidence_depth = 64;
  static constexpr std::size_t query_depth = 24;
  static constexpr std::size_t query_work = 2048;
  static constexpr std::size_t normalization_terms = 256;
  static constexpr std::size_t extent_bound_steps = 8;
  static constexpr std::size_t general_inference_work = 100000;
};

enum class FailureKind { Unsupported, BudgetExceeded, InvalidCandidate };

// Only expected inference failures become Unknown. Programming errors retain
// their original exception type and are not mistaken for unsupported rules.
class ProofFailure : public std::runtime_error {
 public:
  explicit ProofFailure(std::string message,
                        FailureKind kind = FailureKind::Unsupported)
      : std::runtime_error(std::move(message)), kind_(kind) {}
  FailureKind kind() const { return kind_; }
  std::string describe() const {
    switch (kind_) {
      case FailureKind::BudgetExceeded:
        return std::string("budget exhausted: ") + what();
      case FailureKind::InvalidCandidate:
        return std::string("invalid candidate: ") + what();
      case FailureKind::Unsupported:
        return what();
    }
    return what();
  }

 private:
  FailureKind kind_;
};

}  // namespace tepl::core::index_semantics
