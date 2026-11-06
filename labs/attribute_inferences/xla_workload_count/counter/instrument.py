#!/usr/bin/env python3
"""Add opt-in successful rewrite-site instrumentation without changing headers."""
from pathlib import Path
import re,json
root=Path(__file__).resolve().parents[2]
p=root/'xla/hlo/transforms/simplifiers/algebraic_simplifier.cc'
archive=Path(__file__).resolve().parent/'algebraic_simplifier.original.cc'
if archive.exists():
 original=archive.read_text()
 assert p.read_text()==original or 'SetAlgebraicRewriteCounter' in p.read_text()
else:
 original=p.read_text();archive.write_text(original)
# Mask strings/comments while retaining offsets and newlines.
mask=list(original)
lex=re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
for m in lex.finditer(original):
 for i in range(m.start(),m.end()):
  if mask[i]!='\n':mask[i]=' '
mask=''.join(mask)
names=['ReplaceInstructionIfCompatible','ReplaceWithNewInstruction','ReplaceInstruction','ReplaceWithBitcast','SwapCopyBitcastCopy','MarkAsChanged']
pat=re.compile(r'(?<![\w:.>])('+('|'.join(names))+r')\s*\(')
inserts={};sites=[]
for m in pat.finditer(mask):
 start=m.start();opening=mask.index('(',m.start());depth=1;end=opening+1
 while depth:
  if mask[end]=='(':depth+=1
  elif mask[end]==')':depth-=1
  end+=1
 line=original.count('\n',0,start)+1
 # The sink-broadcast helper rewrites each user directly; its caller only
 # marks visitor state, which must not add a second event for that group.
 if m.group(1)=='MarkAsChanged' and line==5656:continue
 inserts.setdefault(start,[]).append(f'ObserveAlgebraicRewrite({line}, __func__, [&]() {{ return ')
 inserts.setdefault(end,[]).append('; })')
 sites.append(dict(line=line,api=m.group(1)))
# This direct graph edit bypasses the visitor replacement helpers. Count
# each actual rewritten user, rather than just the caller's changed flag.
needle='user->ReplaceAllUsesWith(new_broadcast)'
start=original.index(needle,original.index('TryToSinkBroadcastAfterElementwiseOps(')+100)
# The first occurrence after the call may be the scalar-broadcast branch;
# select the occurrence in the helper's definition.
start=original.index(needle,original.index('AlgebraicSimplifierVisitor::TryToSinkBroadcastAfterElementwiseOps'))
line=original.count('\n',0,start)+1;end=start+len(needle)
inserts.setdefault(start,[]).append(f'ObserveAlgebraicRewrite({line}, __func__, [&]() {{ return ')
inserts.setdefault(end,[]).append('; })');sites.append(dict(line=line,api='ReplaceAllUsesWith'))
parts=[]
for i in range(len(original)+1):
 parts.extend(inserts.get(i,[]))
 if i<len(original):parts.append(original[i])
s=''.join(parts)
s=s.replace('#include <tuple>','#include <tuple>\n#include <type_traits>')
helper=r'''
// Research-only opt-in counter. IDs use lines in the archived original file.
using AlgebraicRewriteCounter = void (*)(int, const char*);
namespace {
AlgebraicRewriteCounter algebraic_rewrite_counter = nullptr;
thread_local int algebraic_rewrite_depth = 0;
template <class F>
auto ObserveAlgebraicRewrite(int line, const char* function, F&& body)
    -> std::invoke_result_t<F> {
  if (!algebraic_rewrite_counter) return body();
  using R = std::invoke_result_t<F>;
  const bool outer = algebraic_rewrite_depth++ == 0;
  if constexpr (std::is_void_v<R>) {
    body();
    --algebraic_rewrite_depth;
    if (outer) algebraic_rewrite_counter(line, function);
  } else {
    auto result = body();
    --algebraic_rewrite_depth;
    bool success;
    if constexpr (std::is_same_v<R, bool>) success = result;
    else if constexpr (std::is_same_v<R, absl::StatusOr<bool>>)
      success = result.ok() && result.value();
    else success = result.ok();
    if (outer && success) algebraic_rewrite_counter(line, function);
    return result;
  }
}
}  // namespace
void SetAlgebraicRewriteCounter(AlgebraicRewriteCounter counter) {
  algebraic_rewrite_counter = counter;
}
'''
s=s.replace('namespace xla {','namespace xla {\n'+helper,1)
anchor='''    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  AlgebraicSimplifierVisitor visitor(options_, this);'''
assert anchor in s
s=s.replace(anchor,anchor.replace('  bool changed = false;', '  if (algebraic_rewrite_counter) algebraic_rewrite_counter(0, "RunImpl");\n  bool changed = false;'),1)
p.write_text(s)
(Path(__file__).parent/'instrumented-sites.json').write_text(json.dumps(sites,indent=2)+'\n')
print('Instrumented',len(sites),'successful commit sites')
