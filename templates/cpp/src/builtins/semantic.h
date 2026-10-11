#pragma once
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#include "error.h"
namespace @TEPL_NAMESPACE@ ::builtins::semantic {
using List = ::std::vector<::std::uint64_t>;
enum class Status { Found, Invalid, Unknown };
struct Solution {
  Status status = Status::Unknown;
  ::std::vector<List> fields;
};
using Solve = const char* (*)(const char*);
inline Solve solver() {
  static Solve function = []() -> Solve {
#if defined(__unix__) || defined(__APPLE__)
    auto path = ::std::getenv("TEPL_SEMANTIC_RUNTIME");
    auto handle = ::dlopen(path ? path : "libtepl_semantic_runtime.so",
                           RTLD_NOW | RTLD_LOCAL);
    return handle ? reinterpret_cast<Solve>(
                        ::dlsym(handle, "tepl_semantic_solve_v1"))
                  : nullptr;
#else
    return nullptr;
#endif
  }();
  return function;
}
inline Solution solve(const ::std::string& plan,
                      const ::std::vector<List>& shapes,
                      const ::std::vector<List>& known,
                      const ::std::vector<List>& pinned) {
  auto function = solver();
  if (!function) return {};
  ::std::ostringstream request;
  request << plan;
  for (const auto* lists : {&shapes, &known, &pinned}) {
    request << lists->size() << ' ';
    for (const auto& list : *lists) {
      request << list.size() << ' ';
      for (auto x : list) request << x << ' ';
    }
  }
  const auto payload = request.str();
  auto response = function(payload.c_str());
  if (!response) return {};
  ::std::istringstream in(response);
  int status;
  ::std::size_t count;
  if (!(in >> status >> count)) return {};
  Solution result;
  result.status = status == 0   ? Status::Found
                  : status == 1 ? Status::Invalid
                                : Status::Unknown;
  result.fields.resize(count);
  for (auto& list : result.fields) {
    ::std::size_t length;
    if (!(in >> length)) return {};
    list.resize(length);
    for (auto& x : list)
      if (!(in >> x)) return {};
  }
  return result;
}
inline BuiltinResult<::std::vector<List>> construct(
    const ::std::string& plan, const ::std::vector<List>& shapes,
    const ::std::vector<List>& known, const ::std::vector<List>& pinned) {
  auto result = solve(plan, shapes, known, pinned);
  if (result.status != Status::Found)
    return BuiltinError(
        result.status == Status::Invalid
            ? "semantic inference invalid at matched metadata"
            : "semantic inference unknown or runtime unavailable");
  return result.fields;
}
}  // namespace @TEPL_NAMESPACE@::builtins::semantic
