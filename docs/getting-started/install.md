# Install TEPL

Build the CLI from source or build a Docker image locally. You can run tensor
graph rewrites without Rust or an e-graph library. The source build needs a
C++20 compiler and Java for ANTLR; Bazel manages the project dependencies.

## Option 1: Build from source (Recommended)

### 🧰 Prerequisites

<details markdown="1">
<summary><strong>👉 Ubuntu / Debian</strong></summary>

Install the system packages below, then install Bazel or Bazelisk separately:

```sh
sudo apt-get update
sudo apt-get install -y build-essential ca-certificates curl python3 default-jdk unzip zip
```

</details>

<details markdown="1">
<summary><strong>👉 macOS</strong></summary>

Install Xcode Command Line Tools, then Bazelisk and Java using Homebrew:

```sh
xcode-select --install
# After installation finishes, with Homebrew installed:
brew install bazelisk openjdk
```

Bazelisk provides `bazel` and selects the version pinned in `.bazelversion`.

</details>

The build and binary checks below are the same on both platforms.

### 🔨 Build

From the repository root, confirm the Bazel version and build TEPL:

```sh
bazel --version
bazel build //:tepl
```

The executable is written to `bazel-bin/tepl`. The first build takes longer while
Bazel downloads and compiles dependencies.

### ✅ Check the binary

From the repository root, display the CLI help and validate the example project:

```sh
bazel-bin/tepl --help
bazel-bin/tepl check examples/projects/transpose
```

The help lists the available commands, including `check`, `rewrite`, and
`generate`. The second command prints the checked program, beginning with `CheckedProgram`. Both commands should
exit with status `0`.

### Install the `tepl` command

From the repository root, copy the executable into `/usr/local/bin` for
system-wide use on Linux or macOS:

```sh
sudo mkdir -p /usr/local/bin
sudo install -m 0755 bazel-bin/tepl /usr/local/bin/tepl
tepl --version
tepl --help
```

Ensure `/usr/local/bin` is on your `PATH`. You can then create and check a
project from any directory:

```sh
tepl init my_project
tepl check my_project
```

After updating the source, repeat the build and install commands.

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
  -v "$PWD:/work" tepl check examples/projects/transpose

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" tepl generate examples/basic/double_negation.tepl \
  --out my_app/src/generated --no-format
```

**Always pass `--no-format` when generating code with Docker.** The image omits
`rustfmt`; format generated Rust on your host if needed.

Run a rewrite with the same mount:

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" tepl rewrite examples/projects/transpose --trace -o /work/result.tepl
```

Keep imported TEPL files inside the mounted project. On Linux, `--user` keeps
generated files owned by you. On macOS and Windows with Docker Desktop, you
can usually omit it; these examples use a POSIX shell.

Generated Rust or C++ applications are built separately on your host.

Next: [your first rewrite](first-rewrite.md).
