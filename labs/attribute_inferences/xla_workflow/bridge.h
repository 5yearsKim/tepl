#ifndef TEPL_XLA_WORKFLOW_BRIDGE_H
#define TEPL_XLA_WORKFLOW_BRIDGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  const int64_t* data;
  size_t size;
} WorkflowList;
// op: 0 opaque captured tensor, 1 transpose, 2 dot. Nodes are topological.
typedef struct {
  int op;
  int f32;
  int64_t original_id;
  WorkflowList shape, permutation, fields[4];
  size_t operands[2], arity;
} WorkflowNode;
typedef struct {
  const WorkflowNode* nodes;
  size_t count, root, rule;
} WorkflowResult;
void* tepl_workflow_load(const char* path, char* error, size_t error_size);
// Results remain valid until the next call. 1 matched, 0 no match, -1 error.
int tepl_workflow_match(void*, const WorkflowNode*, size_t count, size_t root,
                        uint64_t excluded_rules, WorkflowResult*, char*,
                        size_t);
uint64_t tepl_workflow_roots(void*);
size_t tepl_workflow_depth(void*);
void tepl_workflow_destroy(void*);
#ifdef __cplusplus
}
#endif
#endif
