#include <string>

#include "src/core/semantics/early_verification.h"
#include "src/core/semantics/solver.h"
// One solver implementation is shared by generated Rust and C++ modules.
// The result is thread-local and valid until the next call on this thread.
extern "C" const char* tepl_semantic_solve_v1(const char* request) noexcept {
  thread_local std::string result;
  try {
    result = tepl::core::semantics::solveRequest(request ? request : "");
  } catch (...) {
    result = "2 0 ";
  }
  return result.c_str();
}

extern "C" const char* tepl_semantic_verify_v1(const char* request) noexcept {
  thread_local std::string result;
  try {
    result = tepl::core::semantics::verificationRequest(request ? request : "");
  } catch (...) {
    result.clear();
  }
  return result.c_str();
}
