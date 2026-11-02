#!/usr/bin/env python3
"""Freeze and run the full CPU compiler workflow; preserve all workload outcomes."""
from pathlib import Path
import datetime,hashlib,json,os,platform,shutil,statistics,subprocess,sys,tarfile
lab=Path(__file__).resolve().parent
paper=Path('/home/onion/Projects/tepl_related/TEPL-paper')
xla=Path(os.environ.get('XLA_ROOT','/home/onion/Projects/xla'));tepl=lab.parents[1]
out=Path(sys.argv[1]) if len(sys.argv)>1 else paper/'evaluation/xla-workflow'
out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
raw=Path('/tmp/tepl-xla-workflow/inputs');manifest=json.loads((raw/'workloads.json').read_text())
for sub in ['inputs','bin','results','logs','hlo','source']:(out/sub).mkdir(exist_ok=True)
for r in manifest['workloads']:
 source=Path(r['input']);target=out/'inputs'/source.name;shutil.copy2(source,target)
 assert sha(target)==r['sha256'];r['input']='inputs/'+source.name
(out/'workloads.json').write_text(json.dumps(manifest,indent=2)+'\n')
for source,name in [(xla/'bazel-bin/research/tepl_workflow/workflow','workflow'),(tepl/'bazel-bin/labs/xla_workflow/tepl_workflow.so','tepl_workflow.so'),(tepl/'bazel-bin/tepl','tepl')]:
 target=out/'bin'/name;temporary=target.with_suffix(target.suffix+'.tmp');shutil.copy2(source,temporary);temporary.replace(target)
shutil.copy2('/tmp/tepl-xla-workflow/program.json',out/'program.json')
shutil.copy2('/tmp/tepl-xla-workflow/plan.json',out/'plan.json')
shutil.copytree(lab,out/'source/labs/xla_workflow',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for source in ['transpose_folding.cc','transpose_folding.h']:
 shutil.copy2(xla/'xla/service'/source,out/'source'/source)
(out/'xla.patch').write_bytes(subprocess.check_output(['git','diff','--','xla/service/transpose_folding.cc'],cwd=xla))
with tarfile.open(out/'tepl-source.tar.gz','w:gz') as archive:
 for name in ['.bazelrc','.bazelversion','src','runtime','grammar','schema','tools','third_party','examples','docs/references/built-ins.md','BUILD.bazel','MODULE.bazel','MODULE.bazel.lock','LICENSE','labs/xla_workflow']:
  if (tepl/name).exists():archive.add(tepl/name,arcname=name,filter=lambda info:None if any(part in ['__pycache__','target','.git'] for part in Path(info.name).parts) else info)
(out/'tepl.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=tepl))
(out/'requirements.txt').write_text(subprocess.check_output(['uv','pip','freeze','--python','/tmp/tepl-xla-workflow-venv/bin/python'],text=True))
provenance=dict(run_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tepl_checkout=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tepl,text=True).strip(),xla_checkout=subprocess.check_output(['git','rev-parse','HEAD'],cwd=xla,text=True).strip(),cpu=next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),platform=platform.platform(),binaries={p.name:sha(p) for p in (out/'bin').iterdir()},source_model_url='https://raw.githubusercontent.com/google/flax/v0.12.0/examples/lm1b/models.py',source_model_sha256=sha(lab/'flax_lm1b_models.py'),rounds=10,input_seeds=3,cpu_client=dict(asynchronous=False,cpu_device_count=1,max_transpose_threads=1),cpu_lowering='Default parsed XLA CPU debug options; no Eigen or pass-disable override.',graph_comparison='Canonical HLO with full constants, backend configuration and control dependencies; ignore instruction identifiers/names and debug metadata.',compile_program='tepl compile inferred.tepl --core --semantic-policy algebraic -o program.json; tepl validate program.json',comparison=dict(atol=1e-5,rtol=1e-5),timing='One untimed warmup per variant; ten paired rounds with rotating variant order. Includes program loading; excludes parsing, cloning, graph comparison, numerical execution, and the diagnostic hook audit. No match/output cache.',build='bazel build --config=joint_shard -c opt --jobs=12 //research/tepl_workflow:workflow',native_build='bazel build -c opt //labs/xla_workflow:tepl_workflow.so //:tepl')
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
env=dict(os.environ);env['LD_LIBRARY_PATH']='/usr/lib/x86_64-linux-gnu'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
test=subprocess.run([str(out/'bin/workflow'),'--self-test',str(out/'bin/tepl_workflow.so'),str(out/'program.json')],env=env,text=True,capture_output=True)
(out/'logs/self-test.stdout').write_text(test.stdout);(out/'logs/self-test.stderr').write_text(test.stderr)
if test.returncode:raise RuntimeError('HLO adapter self-test failed; see logs')
rows=[]
for workload in manifest['workloads']:
 name=workload['name'];args=[str(out/'bin/workflow'),str(out/workload['input']),str(out/'bin/tepl_workflow.so'),str(out/'program.json'),str(out/'hlo'/name),'10','3']
 print('Running '+name,flush=True)
 try:r=subprocess.run(args,env=env,text=True,capture_output=True,timeout=1800)
 except subprocess.TimeoutExpired as error:
  rows.append(dict(name=name,status='timeout'));(out/'logs'/(name+'.stderr')).write_text(str(error));continue
 (out/'logs'/(name+'.stderr')).write_text(r.stderr);(out/'logs'/(name+'.stdout')).write_text(r.stdout)
 if r.returncode:
  row=dict(name=name,status='failed',exit_code=r.returncode,error=r.stderr[-3000:]);rows.append(row);print(json.dumps(row),flush=True);continue
 result=json.loads(r.stdout);(out/'results'/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 samples=result['samples']
 problem=None
 if len(samples)!=20 or len(result['outputs'])!=3:problem='incomplete measurement'
 for mode in ['baseline','tepl']:
  counts={(s['tepl'],s['native_dot'],s['native_conv'],s['pass_calls']) for s in samples if s['mode']==mode}
  if len(counts)!=1:problem='inconsistent rewrite counts in '+mode
 if not result['pass_graphs_equal'] or not result['final_graphs_equal']:problem='HLO mismatch'
 if problem:
  rows.append(dict(name=name,status='failed',error=problem));print(name+': '+problem,flush=True);continue
 base={s['round']:s['compile_ms'] for s in samples if s['mode']=='baseline'};integrated={s['round']:s['compile_ms'] for s in samples if s['mode']=='tepl'}
 ratios=[100*(integrated[i]/base[i]-1) for i in sorted(base)]
 row=dict(name=name,status='passed',baseline_rewrites=result['baseline_rewrites'],tepl_rewrites=result['tepl_rewrites'],retained_rewrites=result['retained_rewrites'],pass_graphs_equal=result['pass_graphs_equal'],final_graphs_equal=result['final_graphs_equal'],outputs_equal=all(x['equal'] for x in result['outputs']),baseline_compile_ms=dict(median=statistics.median(base.values()),min=min(base.values()),max=max(base.values())),tepl_compile_ms=dict(median=statistics.median(integrated.values()),min=min(integrated.values()),max=max(integrated.values())),paired_overhead_percent=dict(median=statistics.median(ratios),min=min(ratios),max=max(ratios)),program_load_ms=dict(median=statistics.median(s['program_load_ms'] for s in samples if s['mode']=='tepl'),min=min(s['program_load_ms'] for s in samples if s['mode']=='tepl'),max=max(s['program_load_ms'] for s in samples if s['mode']=='tepl')))
 rows.append(row);print(json.dumps(row),flush=True)
 (out/'summary.json').write_text(json.dumps(dict(results=rows),indent=2)+'\n')
(out/'summary.json').write_text(json.dumps(dict(results=rows),indent=2)+'\n')
manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if not all(r['status']=='passed' for r in rows):sys.exit(1)
