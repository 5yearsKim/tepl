#!/usr/bin/env python3
"""Exercise generic matching, guarded rejection, and paired axis transport."""
import ctypes as c,itertools,json,sys
class List(c.Structure):_fields_=[('data',c.POINTER(c.c_int64)),('size',c.c_size_t)]
class Node(c.Structure):_fields_=[('op',c.c_int),('f32',c.c_int),('original_id',c.c_int64),('shape',List),('permutation',List),('fields',List*4),('operands',c.c_size_t*2),('arity',c.c_size_t)]
class Result(c.Structure):_fields_=[('nodes',c.POINTER(Node)),('count',c.c_size_t),('root',c.c_size_t),('rule',c.c_size_t)]
lib=c.CDLL(sys.argv[1]);lib.tepl_workflow_load.restype=c.c_void_p;lib.tepl_workflow_load.argtypes=[c.c_char_p,c.c_char_p,c.c_size_t];lib.tepl_workflow_match.argtypes=[c.c_void_p,c.POINTER(Node),c.c_size_t,c.c_size_t,c.c_uint64,c.POINTER(Result),c.c_char_p,c.c_size_t];lib.tepl_workflow_destroy.argtypes=[c.c_void_p]
e=c.create_string_buffer(4096);ctx=lib.tepl_workflow_load(sys.argv[2].encode(),e,len(e));assert ctx,e.value
keep=[]
def xs(v):
 a=(c.c_int64*len(v))(*v);keep.append(a);return List(a,len(v))
def values(v):return list(v.data[:v.size])
def graph(rank,first,second,side,extent):
 keep.clear();dims=[2]*rank;dims[first]=extent;dims[second]=5;p=list(range(rank));p[first],p[second]=p[second],p[first];batch=[i for i in range(rank) if i not in [first,second]]
 ns=(Node*4)()
 for i in [0,1]:ns[i].op=0;ns[i].f32=1;ns[i].original_id=i;ns[i].shape=xs(dims)
 ns[2].op=1;ns[2].f32=1;ns[2].original_id=2;ns[2].shape=xs([dims[i] for i in p]);ns[2].permutation=xs(p);ns[2].operands[0]=side;ns[2].arity=1
 ns[3].op=2;ns[3].f32=1;ns[3].original_id=3;ns[3].shape=xs([2]*len(batch)+[5,5]);ns[3].arity=2;ns[3].operands[:]=[0,2] if side else [2,1]
 fields=[[first],[first],batch,batch];fields[side]=[second]
 for i,f in enumerate(fields):ns[3].fields[i]=xs(f)
 return ns,[[first],[first],batch,batch]
checked=0
for rank in range(2,7):
 for first,second in itertools.combinations(range(rank),2):
  for side in [0,1]:
   for extent in [0,1,3]:
    ns,expected=graph(rank,first,second,side,extent);r=Result();code=lib.tepl_workflow_match(ctx,ns,4,3,0,c.byref(r),e,len(e));assert code==1,(rank,side,e.value)
    node=r.nodes[r.root];assert [values(f) for f in node.fields]==expected
    assert [r.nodes[i].original_id for i in node.operands]==[0,1]
    assert lib.tepl_workflow_match(ctx,ns,4,3,1<<r.rule,c.byref(Result()),e,len(e))==0
    checked+=2
# No transpose; wrong dtype; identity permutation; inconsistent source metadata.
ns,_=graph(2,0,1,0,3);ns[3].operands[:]=[0,1]
assert lib.tepl_workflow_match(ctx,ns,4,3,0,c.byref(Result()),e,len(e))==0
ns,_=graph(2,0,1,0,3);ns[0].f32=0
assert lib.tepl_workflow_match(ctx,ns,4,3,0,c.byref(Result()),e,len(e))==0
ns,_=graph(2,0,1,0,3);ns[2].permutation=xs([0,1]);ns[2].shape=ns[0].shape;ns[3].fields[0]=xs([0])
assert lib.tepl_workflow_match(ctx,ns,4,3,0,c.byref(Result()),e,len(e))==0
ns,_=graph(2,0,1,0,3);ns[3].fields[1]=xs([1])
assert lib.tepl_workflow_match(ctx,ns,4,3,0,c.byref(Result()),e,len(e))==-1
lib.tepl_workflow_destroy(ctx)
print(json.dumps(dict(transport_and_exclusion_checks=checked,rejection_controls=4)))
