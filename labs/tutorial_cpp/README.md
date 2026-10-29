# C++ egg-c tutorial

This example composes two transposes, synthesizes the resulting permutation, and
removes the identity transpose. Generated code uses the shared C++20 runtime in
`runtime/cpp`; egg-c and the runtime are separate dependencies of the application.

```sh
bazel build //:tepl
cmake -S labs/tutorial_cpp -B /tmp/tepl-tutorial-cpp
cmake --build /tmp/tepl-tutorial-cpp
ctest --test-dir /tmp/tepl-tutorial-cpp --output-on-failure
```

CMake downloads egg-c revision `0c28bd5050b85ed10e915b5348c27f760ed31ae5`, verified
by SHA-256. Set `-DEGGC_SOURCE_DIR=/path/to/egg-c` to use a local checkout.
GCC and Clang with signed `__int128` support are supported; MSVC is deferred.

`program.tepl` supplies dialect contracts, rules, and a concrete graph. Generation
writes into the CMake build directory. `src/main.cpp` builds the graph, runs
`rules::all()` through the checked runner, verifies shape/dtype preservation, and
extracts the input with cost one. Application failures are returned as typed
`EvalResult` values; `.value()` propagates their `EvalError` when using exceptions.

The retired structural-only/custom-analysis and host-function APIs have been
removed. The current API follows the Rust generated/runtime boundary.
