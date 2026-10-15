#!/usr/bin/env python3
"""Generate the explicit v2 JSON Schema and its C++ codec from one field catalog.

Generated files are checked in; CLI tests compare the public and embedded schemas.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = {}
E = {}
V = {}
def struct(name, cpp, fields, init=None):
    D[name] = (cpp, dict(x.split(':',1) for x in fields.split()), init)
def enum(name, cpp, values):
    E[name] = (cpp, dict(x.split(':',1) for x in values.split()))
def variant(name, cpp, values):
    V[name] = (cpp, dict(x.split(':',1) for x in values.split()))

enum('dtype','core::DType',' '.join(f'{s}:k{c}' for s,c in [('bool','Bool'),('i8','I8'),('i16','I16'),('i32','I32'),('i64','I64'),('u8','U8'),('u16','U16'),('u32','U32'),('u64','U64'),('f16','F16'),('bf16','BF16'),('f32','F32'),('f64','F64')]))
enum('type_kind','core::TypeKind','tensor:kTensor index:kIndex index_list:kIndexList bool:kBool i64:kI64 f64:kF64 descriptor:kDescriptor dtype:kDType string:kString precision:kPrecision opaque:kOpaque')
enum('descriptor_kind','core::completed::Descriptor::Kind','captured:kCaptured derived:kDerived')
enum('shape_kind','core::ShapeElement::Kind','dimension:kDimension wildcard:kWildcard sequence:kSequence')
enum('literal_kind','core::GraphLiteral::Kind','integer:kInteger decimal:kDecimal')
enum('attribute_kind','core::AttributeValue::Kind','integer:kInteger bool:kBool string:kString enum:kEnum list:kList none:kNone')
enum('node_kind','core::ConcreteNode::Kind','input:kInput literal:kLiteral operation:kOperation')
enum('unary','UnaryOp','+:kPlus -:kNegate !:kLogicalNot')
enum('binary','BinaryOp','+:kAdd -:kSubtract *:kMultiply /:kDivide %:kRemainder <:kLess <=:kLessEqual >:kGreater >=:kGreaterEqual ==:kEqual !=:kNotEqual &&:kLogicalAnd ||:kLogicalOr')
# Public names and IDs belong to the builtin catalog, including in the wire schema.
builtins = re.findall(
    r'^TEPL_BUILTIN\(\s*(k\w+),\s*"([^"]+)"',
    (ROOT / 'src/shared/builtins/catalog.def').read_text(),
    re.MULTILINE,
)
if not builtins:
    raise RuntimeError('empty builtin catalog')
enum('builtin', 'core::builtins::Builtin',
     ' '.join(f'{name}:{identity}' for identity, name in builtins))
enum('meta_kind','core::metadata::Type::Kind','integer:kInteger bool:kBoolean dtype:kDType')
enum('meta_program_kind','core::metadata::ProgramKind','shape:Shape dtype:DType')
enum('axis_meta_kind','core::axis::Meta::Kind','integer:Integer bool:Boolean operand:Operand attribute:Attribute group:Group list:List call:Call')
enum('axis_type','core::axis::Type','integer:Integer bool:Boolean positions:Positions axes:Axes operand:Operand')
enum('symbol_kind','core::completed::Symbol::Kind','integer:Integer bool:Boolean shape:Shape dtype:DType field:Field axis_group:AxisGroup list:List call:Call')
enum('symbol_sort','core::completed::Symbol::Sort','integer:Integer bool:Boolean positions:Positions axes:Axes extents:Extents dtype:DType')
struct('position','SourcePosition','line:id column:id')
struct('span','SourceSpan','begin:position end:position')
struct('location','SourceLocation','source_name:string span:span')
struct('origin','SourceOrigin','definition:location expansions:location[]')
struct('type','core::Type','kind:type_kind schema:id? list_depth:id optional:bool opaque_name:string')
struct('field','core::AttributeField','name:string type:string list_depth:id optional:bool empty_default:bool origin:origin')
struct('schema','core::AttributeSchema','id:id dialect:string name:string fields:field[] origin:origin')
struct('operand','core::Operand','name:string type:id variadic:bool origin:origin')
struct('operation','Operation','id:id dialect:string name:string alias:string? operands:operand[] result:id attributes:id? origin:origin shape:meta_program? dtype:meta_program? axis_contract:axis_definition*?')
struct('capture','core::Capture','id:id name:string type:id origin:origin')
struct('dimension','core::DimensionSymbol','id:id name:string sequence:bool type:id origin:origin')
struct('descriptor','core::completed::Descriptor','id:id name:string kind:descriptor_kind type:id origin:origin')
struct('shape_element','core::ShapeElement','kind:shape_kind symbol:id? origin:origin')
struct('dtype_variable','core::DTypeVariable','id:id name:string type:id origin:origin')
variant('dtype_constraint','core::DTypeConstraint','fixed:dtype variable:dtype_id')
struct('constraint','core::CaptureConstraint','capture:id dtype:dtype_constraint? shape:shape_element[] origin:origin')
struct('literal','core::GraphLiteral','kind:literal_kind spelling:string dtype:dtype?')
struct('capture_pattern','core::CapturePattern','capture:id')
struct('match_operation','core::MatchOperation','operation:id descriptor:id? operands:pattern*[]')
struct('bind_pattern','core::BindPattern','capture:id expression:pattern*')
variant('pattern_value','decltype(core::Pattern::value)','capture:capture_pattern literal:literal operation:match_operation bind:bind_pattern')
struct('pattern','core::Pattern','origin:origin type:id value:pattern_value')
struct('capture_ref','core::CaptureRef','capture:id')
struct('build_operation','core::BuildOperation','operation:id descriptor:id? operands:build*[]')
variant('build_value','decltype(core::BuildExpr::value)','capture:capture_ref literal:literal operation:build_operation')
struct('build','core::BuildExpr','origin:origin type:id value:build_value')
struct('dimension_ref','core::DimensionRef','dimension:id')
struct('descriptor_ref','core::DescriptorRef','descriptor:id')
struct('dtype_ref','core::DTypeVariableRef','variable:id')
struct('field_ref','core::DescriptorFieldRef','descriptor:id schema:id field:id')
struct('number','core::NumericConstant','spelling:string decimal:bool')
struct('call','core::completed::BuiltinCall','builtin:builtin arguments:expr*[]')
struct('unary_expr','core::completed::UnaryExpr','op:unary operand:expr*')
struct('binary_expr','core::completed::BinaryExpr','op:binary lhs:expr* rhs:expr*')
struct('local_ref','core::LocalRef','local:id')
struct('local_let','core::completed::LocalLet','local:id value:expr*')
struct('list_expr','core::completed::ListExpr','elements:expr*[]')
struct('index_expr','core::completed::IndexExpr','value:expr* index:expr*')
struct('conditional','core::completed::ConditionalExpr','condition:expr* then_value:expr* else_value:expr*')
struct('comprehension','core::completed::ComprehensionExpr','element:expr* variable:id iterable:expr*')
struct('string_constant','core::StringConstant','value:string')
struct('enum_constant','core::EnumConstant','value:string')
struct('none','core::NoneConstant','')
struct('optional_value','core::completed::OptionalValue','value:expr*')
struct('descriptor_construct','core::completed::DescriptorConstruct','schema:id fields:expr*[]')
struct('symbolic','core::completed::SymbolicExpr','value:symbol*')
variant('expr_value','decltype(core::completed::TypedExpr::value)','capture:capture_ref dimension:dimension_ref descriptor:descriptor_ref field:field_ref dtype_variable:dtype_ref dtype:dtype number:number bool:bool builtin:call unary:unary_expr binary:binary_expr local:local_ref let:local_let list:list_expr index:index_expr if:conditional comprehension:comprehension string:string_constant enum:enum_constant none:none optional:optional_value construct:descriptor_construct symbolic:symbolic')
struct('expr','core::completed::TypedExpr','origin:origin type:id value:expr_value')
struct('local','core::Local','id:id name:string type:id')
struct('derivation','core::completed::Derivation','target:id? value:expr* origin:origin')
struct('rule','Rule','id:id name:string source_name:string origin:origin captures:capture[] dimensions:dimension[] descriptors:descriptor[] lhs:pattern* constraints:constraint[] conditions:expr*[] locals:local[] derivations:derivation[] rhs:build* dtypes:dtype_variable[]')
struct('attribute','core::AttributeValue','kind:attribute_kind text:string elements:attribute[]')
struct('graph_node','core::ConcreteNode','kind:node_kind origin:origin text:string dtype:dtype? shape:extent[]? operation:id? operands:id[] attributes:attribute[]')
struct('graph','core::ConcreteGraph','id:id name:string source_name:string origin:origin nodes:graph_node[] bindings:binding[] root:id')
struct('dialect','core::Dialect','name:string origin:origin')
struct('ir','ProgramData','dialects:dialect[] sources:string[] types:type[] operations:operation[] attribute_schemas:schema[] rules:rule[] graphs:graph[]')
struct('meta_type','core::metadata::Type','kind:meta_kind list_depth:id')
struct('meta_symbol_ref','core::metadata::SymbolRef','symbol:id')
struct('meta_integer','core::metadata::Integer','spelling:string')
struct('meta_attribute','core::metadata::AttributeRef','schema:id field:id')
struct('meta_call','core::metadata::Call','builtin:builtin arguments:meta_expr*[]')
struct('meta_unary','core::metadata::Unary','op:unary operand:meta_expr*')
struct('meta_binary','core::metadata::Binary','op:binary lhs:meta_expr* rhs:meta_expr*')
struct('meta_list','core::metadata::List','elements:meta_expr*[]')
struct('meta_index','core::metadata::Index','value:meta_expr* index:meta_expr*')
struct('meta_conditional','core::metadata::Conditional','condition:meta_expr* then_value:meta_expr* else_value:meta_expr*')
struct('meta_comprehension','core::metadata::Comprehension','element:meta_expr* variable:id iterable:meta_expr*')
variant('meta_expr_value','decltype(core::metadata::Expr::value)','symbol:meta_symbol_ref integer:meta_integer bool:bool dtype:dtype attribute:meta_attribute builtin:meta_call unary:meta_unary binary:meta_binary list:meta_list index:meta_index if:meta_conditional comprehension:meta_comprehension')
struct('meta_expr','core::metadata::Expr','origin:origin type:id value:meta_expr_value')
struct('meta_symbol','core::metadata::Symbol','id:id name:string type:id origin:origin')
struct('parameter','core::metadata::Parameter','symbol:id operand:id variadic:bool')
struct('meta_let','core::metadata::Let','symbol:id value:meta_expr*')
struct('meta_assert','core::metadata::Assert','condition:meta_expr*')
variant('statement_value','decltype(core::metadata::Statement::value)','let:meta_let assert:meta_assert')
struct('statement','core::metadata::Statement','origin:origin value:statement_value')
struct('yield','core::metadata::Yield','origin:origin value:meta_expr*')
struct('meta_program','core::metadata::Program','origin:origin types:meta_type[] symbols:meta_symbol[] parameters:parameter[] statements:statement[] result:yield kind:meta_program_kind')
struct('axis_meta','core::axis::Meta','kind:axis_meta_kind type:axis_type name:string index:id args:axis_meta*[] origin:origin')
struct('axis_binding','core::axis::Binding','operand:id positions:axis_meta*')
struct('axis_group','core::axis::Group','name:string bindings:axis_binding[] origin:origin domain:axis_meta*?')
struct('axis_definition','core::axis::ShapeContract','operands:string[] fields:string[] groups:axis_group[] assertions:axis_meta*[] output:axis_meta* extent_fields:idset implicit_operands:idset')
struct('symbol','core::completed::Symbol','kind:symbol_kind name:string owner:id index:id args:symbol*[] sort:symbol_sort','core::completed::Symbol::Kind::Integer')

def ref(t):
    if t.endswith('?'): return {'anyOf':[ref(t[:-1]),{'type':'null'}]}
    if t.endswith('[]'): return {'type':'array','items':ref(t[:-2])}
    if t.endswith('*'): return ref(t[:-1])
    return {'$ref':'#/$defs/'+t}
def obj(props): return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}
def cpp(t):
    if t in D:return D[t][0]
    if t in E:return E[t][0]
    if t in V:return V[t][0]
    return {'bool':'bool','dtype_id':'core::DTypeVariableId'}[t]
defs={'extent':{'type':'string','pattern':'^(0|[1-9][0-9]*)$'},'signed_integer':{'type':'string','pattern':'^-?(0|[1-9][0-9]*)$'},'id':{'type':'integer','minimum':0,'maximum':9007199254740991},'integer':{'type':'integer'},'string':{'type':'string'},'bool':{'type':'boolean'},'dtype_id':ref('id'),'idset':{'type':'array','items':ref('id'),'uniqueItems':True},'binding':obj({'name':ref('string'),'node':ref('id')})}
code=['// Generated by tools/build_program_schema.py. Do not edit.', 'std::uint64_t parseExtent(const std::string&);', 'std::int64_t parseSigned(const std::string&);']
for name,(typ,values) in E.items():
    defs[name]={'enum':list(values)}
    code+=['template<> struct Codec<'+typ+'> {', 'static Json pack('+typ+' v) { switch(v) {']
    code += ['case '+typ+'::'+val+': return '+json.dumps(key)+';' for key,val in values.items()]
    code += ['} throw std::invalid_argument("invalid '+name+'"); }', 'static '+typ+' unpack(const Json& j) {']
    code += ['if(j == '+json.dumps(key)+') return '+typ+'::'+val+';' for key,val in values.items()]
    code += ['throw std::invalid_argument("invalid '+name+'"); }};']
for name,(typ,fields,init) in D.items():
    defs[name]=obj({key:ref(t) for key,t in fields.items()})
    code += ['template<> struct Codec<'+typ+'> { static Json pack(const '+typ+'&); static '+typ+' unpack(const Json&); };']
for name,(typ,values) in V.items():
    defs[name]={'oneOf':[obj({'kind':{'const':tag},'data':ref(t)}) for tag,t in values.items()]}
    code+=['template<> struct Codec<'+typ+'> { static Json pack(const '+typ+'&); static '+typ+' unpack(const Json&); };']
for name,(typ,values) in V.items():
    code += ['inline Json Codec<'+typ+'>::pack(const '+typ+'& v) { return std::visit([](const auto& x)->Json { using T=std::decay_t<decltype(x)>;']
    for tag,t in values.items(): code+=['if constexpr(std::is_same_v<T,'+cpp(t)+'>) return Json{{"kind",'+json.dumps(tag)+'},{"data",detail::pack(x)}};']
    code += ['throw std::invalid_argument("invalid variant"); }, v); }','inline '+typ+' Codec<'+typ+'>::unpack(const Json& j) {']
    for tag,t in values.items(): code+=['if(j.at("kind") == '+json.dumps(tag)+') return detail::unpack<'+cpp(t)+'>(j.at("data"));']
    code+=['throw std::invalid_argument("invalid variant"); }']
for name,(typ,fields,init) in D.items():
    code+=['inline Json Codec<'+typ+'>::pack(const '+typ+'& v) { Json j=Json::object();']
    for key in fields:
        if name=='graph_node' and key=='shape':
            code+=['j["shape"]=nullptr; if(v.shape){j["shape"]=Json::array();for(auto d:*v.shape)j["shape"].push_back(std::to_string(d));}']
        elif name=='axis_value' and key=='constant': code+=['j["constant"]=std::to_string(v.constant);']
        else: code+=['j["'+key+'"] = detail::pack(v.'+key+');']
    code+=['return j; }','inline '+typ+' Codec<'+typ+'>::unpack(const Json& j) { '+typ+' v{'+(init or '')+'};']
    for key in fields:
        if name=='graph_node' and key=='shape':
            code+=['if(!j.at("shape").is_null()){v.shape.emplace();for(const auto& d:j.at("shape"))v.shape->push_back(parseExtent(d.get<std::string>()));}']
        elif name=='axis_value' and key=='constant': code+=['v.constant=parseSigned(j.at("constant").get<std::string>());']
        else: code+=['v.'+key+' = detail::unpack<decltype(v.'+key+')>(j.at("'+key+'"));']
    code+=['return v; }']
# std::set codec declared before generated struct bodies.
header=(ROOT/'src/executable/codec.h').read_text().split('// Generated by tools/build_program_schema.py.')[0]
if 'Codec<std::set<T>>' not in header:
    header += '''template<class T> struct Codec<std::set<T>> {
  static Json pack(const std::set<T>& vs) { return detail::pack(std::vector<T>(vs.begin(),vs.end())); }
  static std::set<T> unpack(const Json& j) { auto v=detail::unpack<std::vector<T>>(j); return {v.begin(),v.end()}; }
};
'''
(ROOT/'src/executable/codec.h').write_text(header+'\n'.join(code)+'\n} // namespace tepl::executable::detail\n')
schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'TEPL compiled rewrite program v2',**obj({'format':{'const':'tepl.rewrite'},'version':{'const':2},'policy':{'enum':['strict','algebraic']},'capabilities':{'const':['tepl.core.v2','tepl.axis-metadata.v2']},'program':ref('ir'),'axis_definitions':ref('symbol*[][]')}),'$defs':defs}
text=json.dumps(schema,indent=2)+'\n'
(ROOT/'schema/rewrite-program.schema.json').write_text(text)
(ROOT/'src/executable/schema_data.h').write_text('// Generated by tools/build_program_schema.py. Do not edit.\n#pragma once\n#include <string_view>\nnamespace tepl::executable::detail {\ninline constexpr std::string_view kSchema = R"TEPL_SCHEMA('+text+')TEPL_SCHEMA";\n}\n')
