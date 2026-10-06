# 🧰 Set up TEPL for the Rust tutorial

Build TEPL from source (recommended) or build a Docker image locally.

## Option 1: Build from source (Recommended)

### 🧰 Prerequisites

<details markdown="1">
<summary><strong>👉 Ubuntu / Debian</strong></summary>

Install the system packages below, then install Bazel or Bazelisk separately:

```sh
sudo apt-get update
sudo apt-get install -y build-essential ca-certificates curl python3 unzip zip
```

</details>

<details markdown="1">
<summary><strong>👉 macOS</strong></summary>

Install Xcode Command Line Tools, then Bazelisk using Homebrew:

```sh
xcode-select --install
# After installation finishes, with Homebrew installed:
brew install bazelisk
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
bazel-bin/tepl check examples/sample
```

The help lists the `parse`, `check`, and `generate` commands. The second command
prints the checked program, beginning with `CheckedProgram`. Both commands should
exit with status `0`.

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
  --out my_app/src/generated --no-format
```

**Always pass `--no-format` when generating code with Docker.** The image omits
`rustfmt`; format generated Rust on your host if needed.

Keep imported TEPL files inside the mounted project. On Linux, `--user` keeps
generated files owned by you. On macOS and Windows with Docker Desktop, you
can usually omit it; these examples use a POSIX shell.

Build and run the generated Rust with Cargo on your host.

Next: [write and run the basic example](02_run.md).
