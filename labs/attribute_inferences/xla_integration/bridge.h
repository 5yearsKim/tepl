#ifndef TEPL_XLA_BRIDGE_H
#define TEPL_XLA_BRIDGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  const int64_t* data;
  size_t size;
} TeplList;
// Opaque C boundary: no C++ containers or exceptions cross compiler toolchains.
void* tepl_xla_load(const char* inferred, char* error, size_t error_size);
// Apply the accepted inferred program to one selected operand.
// fields: lhs-contract, rhs-contract, lhs-batch, rhs-batch.
int tepl_xla_fold(void* context, int side, TeplList lhs_shape,
                  TeplList rhs_shape, TeplList permutation,
                  const TeplList fields[4], int64_t* output, size_t capacity,
                  size_t lengths[4], char* error, size_t error_size);
// Clear cached entries on request; counters are monotone across clears.
int tepl_xla_cache(void* context, int clear, uint64_t* hits, uint64_t* misses);
void tepl_xla_destroy(void* context);
#ifdef __cplusplus
}
#endif
#endif
