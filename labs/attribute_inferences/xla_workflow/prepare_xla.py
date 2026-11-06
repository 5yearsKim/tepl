#!/usr/bin/env python3
"""Install opt-in instrumentation; default behavior remains upstream behavior."""
from pathlib import Path
import shutil,sys
lab=Path(__file__).resolve().parent
xla=Path(sys.argv[1] if len(sys.argv)>1 else '/home/onion/Projects/xla')
cc=xla/'xla/service/transpose_folding.cc'
s=cc.read_text()
if 'SetTransposeFoldingWorkflowHook' not in s:
 (lab/'transpose_folding.upstream.cc').write_text(s)
 anchor='absl::StatusOr<bool> TransposeFolding::RunImpl('
 block='''using WorkflowHook = absl::StatusOr<bool>(*)(HloModule*,const absl::flat_hash_set<absl::string_view>&,const TransposeFolding::CanFoldTransposeOperand&,const TransposeFolding::TransposableConvOperandsFn&);
using WorkflowObserver = void(*)(int,size_t);
namespace { WorkflowHook workflow_hook=nullptr; WorkflowObserver workflow_observer=nullptr; }
void SetTransposeFoldingWorkflowHook(WorkflowHook hook){workflow_hook=hook;}
void SetTransposeFoldingWorkflowObserver(WorkflowObserver observer){workflow_observer=observer;}

'''
 assert anchor in s;s=s.replace(anchor,block+anchor,1)
 anchor='''    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Modifying'''
 replacement='''    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if(workflow_hook)return workflow_hook(module,execution_threads,dot_can_fold_transpose_operand_,transposable_conv_operands_);
  // Modifying'''
 assert anchor in s;s=s.replace(anchor,replacement,1)
 anchor='''    ABSL_RETURN_IF_ERROR(FoldTransposeIntoDot(pair));
    changed = true;'''
 assert anchor in s;s=s.replace(anchor,'''    ABSL_RETURN_IF_ERROR(FoldTransposeIntoDot(pair));
    if(workflow_observer)workflow_observer(0,pair.second.size());
    changed = true;''',1)
 anchor='''    changed |= FoldTransposeIntoConvolution(pair);'''
 assert anchor in s;s=s.replace(anchor,'''    bool folded=FoldTransposeIntoConvolution(pair);
    if(folded&&workflow_observer)workflow_observer(1,pair.second.size());
    changed |= folded;''',1)
 cc.write_text(s)
out=xla/'research/tepl_workflow';out.mkdir(parents=True,exist_ok=True)
for name in ['driver.cc','hook.h','bridge.h','xla.BUILD.bazel']:
 shutil.copy2(lab/name,out/('BUILD.bazel' if name=='xla.BUILD.bazel' else name))
print(out)
