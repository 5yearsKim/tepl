#ifndef TEPL_XLA_WORKFLOW_HOOK_H
#define TEPL_XLA_WORKFLOW_HOOK_H
#include "xla/service/transpose_folding.h"
namespace xla {
using WorkflowHook = absl::StatusOr<bool> (*)(
    HloModule*, const absl::flat_hash_set<absl::string_view>&,
    const TransposeFolding::CanFoldTransposeOperand&,
    const TransposeFolding::TransposableConvOperandsFn&);
using WorkflowObserver = void (*)(int, size_t);
void SetTransposeFoldingWorkflowHook(WorkflowHook);
void SetTransposeFoldingWorkflowObserver(WorkflowObserver);
}  // namespace xla
#endif
