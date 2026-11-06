# Set up egg-c integration

First [install TEPL](../../../getting-started/install.md). Read
[how e-graphs work](../overview.md) for the search and extraction model.

C++ applications need GCC or Clang with signed `__int128` support, the shared
`runtime/cpp` headers, and egg-c. CMake 3.20 or later and Git/curl are useful for
downloading dependencies. The TEPL compiler itself does not depend on egg-c.

Check and generate the tutorial:

```sh
bazel-bin/tepl check labs/tutorial_cpp/program.tepl
bazel-bin/tepl generate labs/tutorial_cpp/program.tepl \
  --target cpp --cpp-namespace tutorial_ir --out /tmp/tutorial-generated
```

`clang-format` formats generated C++ by default. Install it or use `--no-format`.
The output is a relocatable set of program-specific headers; include
`generated.h` in the application and link the `tepl::runtime` CMake interface
target. MSVC support is deferred.

The [next guide](run.md) builds the complete CMake example, including its pinned
egg-c dependency and generated headers.
