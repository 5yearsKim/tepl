#ifndef TEPL_XLA_ALGEBRAIC_BRIDGE_H
#define TEPL_XLA_ALGEBRAIC_BRIDGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  const int64_t* data;
  size_t size;
} AlgebraicList;
// op: 0 opaque captured tensor, 1 transpose, 2 dot. Nodes are topological.
typedef struct {
  int op;
  int f32;
  int64_t original_id;
  AlgebraicList shape, permutation, fields[4];
  size_t operands[2], arity;
} AlgebraicNode;
typedef struct {
  const AlgebraicNode* nodes;
  size_t count, root, rule;
} AlgebraicResult;
void* tepl_algebraic_load(const char* path, char* error, size_t error_size);
// Results remain valid until the next call. 1 matched, 0 no match, -1 error.
int tepl_algebraic_match(void*, const AlgebraicNode*, size_t count, size_t root,
                         uint64_t excluded_rules, AlgebraicResult*, char*,
                         size_t);
uint64_t tepl_algebraic_roots(void*);
size_t tepl_algebraic_depth(void*);
void tepl_algebraic_destroy(void*);
// Versioned branch protocol. JSON owns typed attributes and arbitrary operand
// lists; returned bytes remain valid until the next apply on this context.
void* tepl_branch_load(const char*, char*, size_t);
int tepl_branch_apply(void*, const char*, const char**, char*, size_t);
void tepl_branch_destroy(void*);
#ifdef __cplusplus
}
#endif
#endif
