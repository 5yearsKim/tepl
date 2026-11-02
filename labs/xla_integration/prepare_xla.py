#!/usr/bin/env python3
"""Pin the complete upstream pass test inventory, retaining unsupported cases."""
from pathlib import Path
import re,json,hashlib,shutil,os
HERE=Path(__file__).resolve().parent
XLA=Path(os.environ.get('XLA_ROOT','/home/onion/Projects/xla'))
OUT=HERE/'inputs';OUT.mkdir(exist_ok=True)
source=XLA/'xla/service/transpose_folding_test.cc';text=source.read_text()
parts=re.split(r'TEST_F\(TransposeFoldingTest,\s*(\w+)\)\s*\{',text)
manifest=[];builders=[]
for name,body in zip(parts[1::2],parts[2::2]):
 raws=re.findall(r'R"([\w]*)\((.*?)\)\1"',body,re.S)
 row={'name':name,'config':'custom' if name=='FoldTransposeOfBatchWhenPermitted' else 'default','family':'convolution' if 'Conv' in name else 'dot_or_control','source_test_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
 if raws:
  assert len(raws)==1
  hlo=raws[0][1];(OUT/f'{name}.hlo').write_text(hlo);row['input']='literal';row['sha256']=hashlib.sha256(hlo.encode()).hexdigest()
 else:
  prefix=re.split(r'\bEXPECT_',body,maxsplit=1)[0]
  assert 'auto module = CreateNewVerifiedModule' in prefix,name
  prefix=prefix.replace('ASSERT_OK_AND_ASSIGN','ABSL_ASSIGN_OR_RETURN')
  prefix+='\n (void)entry_computation;\n'
  if name=='FuseDotWithConstantOperands':prefix+=' (void)call;\n'
  builders.append('absl::StatusOr<std::unique_ptr<HloModule>> make_'+name+'(){\n'+prefix+'return module;\n}\n')
  row['input']='upstream_builder_prefix';row['builder_sha256']=hashlib.sha256(prefix.encode()).hexdigest()
 manifest.append(row)
assert len(manifest)==25
(HERE/'manifest-inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
(HERE/'upstream_builders.inc').write_text('\n'.join(builders)+'\nabsl::StatusOr<std::unique_ptr<HloModule>> makeBuilder(const std::string& name){\n'+''.join(f'if(name=="{r["name"]}")return make_{r["name"]}();\n' for r in manifest if r['input']=='upstream_builder_prefix')+'return absl::InvalidArgumentError("unknown builder");\n}\n')
# Preserve the entire traversal, backend guards, convolution path and clone logic.
cc=(XLA/'xla/service/transpose_folding.cc').read_text().replace('"xla/service/transpose_folding.h"','"research/tepl_integration/transpose_folding.h"').replace('TransposeFolding','TeplTransposeFolding')
cc=cc.replace('namespace xla {','namespace xla {\nabsl::Status FoldDotViaTepl(HloInstruction*, const std::vector<int64_t>&);\nbool UseHandwrittenDot();\nvoid RecordHandwrittenDot();\nvoid RecordHandwrittenConvolution();',1)
cc=cc.replace('absl::Status FoldTransposeIntoDot(InstructionOperandsPair& pair)', 'absl::Status FoldTransposeIntoDotHandwritten(InstructionOperandsPair& pair)')
pos=cc.index('// Folds the operands of `convolution`')
cc=cc[:pos]+'''absl::Status FoldTransposeIntoDot(InstructionOperandsPair& pair){
 if(UseHandwrittenDot()){RecordHandwrittenDot();return FoldTransposeIntoDotHandwritten(pair);}
 return FoldDotViaTepl(pair.first,pair.second);
}

'''+cc[pos:]
cc=cc.replace('  const ConvolutionDimensionNumbers& dnums =', '  RecordHandwrittenConvolution();\n  const ConvolutionDimensionNumbers& dnums =')
(HERE/'transpose_folding.cc').write_text(cc)
h=(XLA/'xla/service/transpose_folding.h').read_text().replace('XLA_SERVICE_TRANSPOSE_FOLDING_H_','TEPL_INTEGRATION_TRANSPOSE_FOLDING_H_').replace('TransposeFolding','TeplTransposeFolding').replace('return "transpose-folding"','return "tepl-transpose-folding"')
(HERE/'transpose_folding.h').write_text(h)
PACKAGE=XLA/'research/tepl_integration';PACKAGE.mkdir(exist_ok=True)
for n in ['bridge.h','transpose_folding.cc','transpose_folding.h','upstream_builders.inc','xla_driver.cc']:
 shutil.copy2(HERE/n,PACKAGE/n)
shutil.copy2(HERE/'xla.BUILD.bazel',PACKAGE/'BUILD.bazel')
print(f'{len(manifest)} upstream cases: {sum(r["input"]=="literal" for r in manifest)} literal modules, {len(builders)} unchanged builder prefixes')
