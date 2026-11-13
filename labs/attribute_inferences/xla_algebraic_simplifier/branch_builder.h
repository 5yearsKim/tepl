#ifndef TEPL_XLA_BRANCH_BUILDER_H_
#define TEPL_XLA_BRANCH_BUILDER_H_

#include <map>
#include <string>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
// Standalone serial evaluation adapter. A new replacement root is returned
// without a parent; XLA's original branch commit takes ownership of it.
class TeplBranchAdapter {
 public:
  TeplBranchAdapter(const char* library, const char* branches,
                    const char* automatic, bool strict);
  ~TeplBranchAdapter();
  TeplBranchAdapter(const TeplBranchAdapter&) = delete;
  TeplBranchAdapter& operator=(const TeplBranchAdapter&) = delete;
  absl::StatusOr<HloInstruction*> Propose(HloInstruction*, int,
                                          absl::Span<HloInstruction* const>,
                                          absl::Span<const int64_t> = {},
                                          absl::Span<const int64_t> = {});
  absl::StatusOr<HloInstruction*> Transpose(HloInstruction*, int);
  void Observe(int site);
  std::map<int, int64_t> proposals, committed;
  static bool IsPorted(int site);

 private:
  absl::StatusOr<HloInstruction*> Apply(HloInstruction*, int,
                                        const std::string&, bool);
  void* library_ = nullptr;
  void* branches_ = nullptr;
  void* automatic_ = nullptr;
  int (*apply_)(void*, const char*, const char**, char*, size_t) = nullptr;
  void (*destroy_)(void*) = nullptr;
  bool strict_;
  std::map<int, bool> pending_;
};
}  // namespace xla
#endif
