# 🧰 Build TEPL for the C++ tutorial

TEPL uses the same compiler for Rust and C++ generation. If you already built
`bazel-bin/tepl`, keep that executable and continue with
[Run your first TEPL code](02_run.md).

## Prerequisites

For this tutorial, you need:

- Bazel or Bazelisk to build TEPL.
- GCC or Clang with C++20 and signed `__int128` support.
- CMake 3.20 or newer to build the application.
- Git to fetch egg-c.

On Ubuntu or Debian, install the system tools, then install Bazel or Bazelisk
separately:

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake git ca-certificates curl python3 unzip zip
```

On macOS, install Xcode Command Line Tools and, with Homebrew available,
Bazelisk and CMake:

```sh
xcode-select --install
# After installation finishes:
brew install bazelisk cmake
```

Bazelisk selects the Bazel version pinned in `.bazelversion`. Generated C++
uses `__int128` for checked shape arithmetic, so MSVC is currently unsupported.
The compiler does not need Rust or Cargo to generate C++.

## Build and check TEPL

From the TEPL repository root:

```sh
bazel --version
bazel build //:tepl
bazel-bin/tepl --help
bazel-bin/tepl check labs/tutorial_cpp/pattern_basic
```

The executable is written to `bazel-bin/tepl`. The first build downloads and
compiles dependencies. The final command prints a program beginning with
`CheckedProgram`; all commands should exit with status `0`.

Add the executable's directory to your `PATH` for this terminal:

```sh
export PATH="$PWD/bazel-bin:$PATH"
tepl --help
```

To make this permanent, add the export to your shell configuration using the
absolute path to the repository's `bazel-bin` directory.

## Where egg-c fits

TEPL generates headers; your C++ application includes those headers and
the [egg-c](https://github.com/5yearsKim/egg-c) library. egg-c is header-only,
so there is no separate library binary to compile or link.

The next guide uses CMake to fetch the tested egg-c revision
`0c28bd5050b85ed10e915b5348c27f760ed31ae5`. That first configuration needs
network access. The compiler's own build does not depend on egg-c.

You can also generate headers using the
[shared Docker build instructions](../rust/01_build.md#build-with-docker).
Pass `--target cpp --no-format` when generating inside that image; it does
not include `clang-format`. Build the resulting C++ application on your host.

Next: [write and run the basic example](02_run.md).
