#!/usr/bin/env python3
"""Render measured workflow results; reject incomplete or mismatched runs."""
from pathlib import Path
import json,sys
artifact=Path(sys.argv[1]);paper=Path(sys.argv[2])
rows=json.loads((artifact/'summary.json').read_text())['results']
assert len(rows)==4 and all(r['status']=='passed' for r in rows)
names={'transformer_b1_l16':('TransformerLM','1'),'transformer_b4_l16':('TransformerLM','4'),'dlrm':('DLRM training','6,272'),'resnet50':('ResNet-50','1')}
main=[r'\begin{table}[!htbp]',r'\caption{Full XLA CPU compilation with the selected transpose/dot rules. Counts are folded operands in \code{TransposeFolding}. HLO compares both the pass output and final optimized graph; values compare three input sets. Compilation change is the median [min,max] of ten paired rounds, including fresh TEPL program loading.}',r'\label{tab:xla-workflow}',r'\small',r'\begin{tabularx}{\linewidth}{@{}Yrrrrr@{}}\toprule',r'Workload & Batch & TEPL / baseline & HLO & Values & Compile change (\%)\\\midrule']
timing=[r'\begin{table}[!htbp]',r'\caption{CPU compilation costs in milliseconds: median [min,max] over ten measured rounds per variant. TEPL program loading is included in integrated compilation.}',r'\label{tab:xla-workflow-timings}',r'\small',r'\begin{tabularx}{\linewidth}{@{}Yrrr@{}}\toprule',r'Workload & Baseline & Integrated & Program load\\\midrule']
def fmt(d):
 return r'\shortstack[r]{'+f"{d['median']:.1f}"+r'\\[-2pt]{\scriptsize['+f"{d['min']:.1f},{d['max']:.1f}"+r']}}' 
for r in rows:
 assert r['pass_graphs_equal'] and r['final_graphs_equal'] and r['outputs_equal']
 assert r['tepl_rewrites']+r['retained_rewrites']==r['baseline_rewrites']
 name,batch=names[r['name']];change=r['paired_overhead_percent']
 main.append(f"{name} & {batch} & {r['tepl_rewrites']} / {r['baseline_rewrites']} & = & 3/3 & {fmt(change)}"+r'\\')
 timing.append(f"{name} ({batch}) & {fmt(r['baseline_compile_ms'])} & {fmt(r['tepl_compile_ms'])} & {fmt(r['program_load_ms'])}"+r'\\')
for lines in [main,timing]:lines.extend([r'\bottomrule\end{tabularx}',r'\end{table}'])
(paper/'sections/xla-workflow-summary.tex').write_text('\n'.join(main)+'\n')
(paper/'sections/xla-workflow-timings.tex').write_text('\n'.join(timing)+'\n')
print(json.dumps(rows,indent=2))
