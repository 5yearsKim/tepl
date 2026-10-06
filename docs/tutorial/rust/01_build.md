## Build from Source

### 🧰 Prerequisites

<details>
<summary><strong>👉 Ubuntu / Debian</strong></summary>

Install the system packages below, then install Bazel or Bazelisk separately:

```sh
sudo apt-get update
sudo apt-get install -y build-essential ca-certificates curl python3 unzip zip
```

</details>

<details>
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

## Build with Docker

Build the image from the repository root:

```sh
docker build -t tepl:local .
docker run --rm tepl:local --help
```

The multi-stage build uses the Bazel version in `.bazelversion` and produces an
optimized, stripped executable. The final image contains TEPL and a Distroless
C++ runtime, runs as non-root, and supports builds for Linux amd64 and arm64.
The first build downloads the compiler dependencies; subsequent builds reuse a
Bazel cache managed by Docker BuildKit.

Mount your project at `/work` so TEPL can read inputs and write generated files.
On Linux, use your user and group IDs to keep generated files owned by you:

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" tepl:local check examples/sample

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" tepl:local generate examples/sample \
  --out my_app/src/generated --no-format

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" tepl:local generate examples/sample --target cpp \
  --out my_cpp_app/generated --no-format
```

**Always pass `--no-format` when generating code in Docker.** The image omits
`rustfmt` and `clang-format` to keep it small. Format the generated files on your
host if needed. Mount the input files together with their imported files so
relative import paths remain valid. On macOS and Windows with Docker Desktop,
you can omit `--user`; the examples above use a POSIX shell.

Inspect the local image size with:

```sh
docker image inspect tepl:local --format '{{.Size}}'
```

The result is in bytes. `docker image ls tepl:local` also shows Docker's local
storage usage; reported sizes depend on the Docker storage backend.

