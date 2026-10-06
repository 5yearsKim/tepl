# 🧰 Set up TEPL for the C++ tutorial

TEPL uses the same compiler for Rust and C++ generation. Build it from
source (recommended) or build a Docker image locally. If you already have
`bazel-bin/tepl`, continue with [Run your first TEPL code](02_run.md).

## Option 1: Build from source (Recommended)

### Prerequisites

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

### Build and check TEPL

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

### Where egg-c fits

TEPL generates headers; your C++ application includes those headers and
the [egg-c](https://github.com/5yearsKim/egg-c) library. egg-c is header-only,
so there is no separate library binary to compile or link.

The next guide uses CMake to fetch the tested egg-c revision
`0c28bd5050b85ed10e915b5348c27f760ed31ae5`. That first configuration needs
network access. The compiler's own build does not depend on egg-c.

## Option 2: Build with Docker

With Docker installed, build the image from the TEPL repository root:

```sh
docker build -t tepl .
docker run --rm tepl --help
```

This builds for your machine's architecture. The first build downloads and
compiles dependencies; later builds reuse the Bazel cache.

Mount your project at `/work` so TEPL can read inputs and write generated files.
From the repository root:

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" tepl check examples/sample

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" tepl generate examples/sample \
  --target cpp --out my_cpp_app/generated --no-format
```

**Always pass `--no-format` when generating code with Docker.** The image omits
`clang-format`; format generated headers on your host if needed.

Keep imported TEPL files inside the mounted project. On Linux, `--user` keeps
generated files owned by you. On macOS and Windows with Docker Desktop, you
can usually omit it; these examples use a POSIX shell.

Build the generated C++ application with GCC or Clang, CMake, and egg-c
on your host.

Next: [write and run the basic example](02_run.md).
