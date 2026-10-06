# syntax=docker/dockerfile:1

FROM debian:trixie-slim AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl python3 unzip zip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY .bazelversion ./
ARG TARGETARCH
RUN set -eu; \
    case "${TARGETARCH}" in \
        amd64) bazel_arch=x86_64 ;; \
        arm64) bazel_arch=arm64 ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    bazel_version="$(cat .bazelversion)"; \
    bazel_file="bazel-${bazel_version}-linux-${bazel_arch}"; \
    bazel_url="https://releases.bazel.build/${bazel_version}/release/${bazel_file}"; \
    curl -fsSL "${bazel_url}" -o "/tmp/${bazel_file}"; \
    curl -fsSL "${bazel_url}.sha256" -o "/tmp/${bazel_file}.sha256"; \
    cd /tmp; \
    sha256sum -c "${bazel_file}.sha256"; \
    install -m 0755 "${bazel_file}" /usr/local/bin/bazel; \
    rm "${bazel_file}" "${bazel_file}.sha256"

COPY .bazelrc BUILD.bazel MODULE.bazel MODULE.bazel.lock ./
COPY grammar/ grammar/
COPY examples/ examples/
COPY src/ src/
COPY templates/ templates/
COPY third_party/ third_party/
COPY tools/ tools/

# Copy the binary out of the cache mount before it disappears at the end of RUN.
RUN --mount=type=cache,id=tepl-bazel-${TARGETARCH},target=/root/.cache/bazel,sharing=locked \
    bazel --batch build -c opt --strip=always //:tepl \
    && install -D -m 0755 bazel-bin/tepl /out/tepl

FROM gcr.io/distroless/cc-debian13:nonroot
COPY --from=build /out/tepl /usr/local/bin/tepl
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/tepl"]
CMD ["--help"]
