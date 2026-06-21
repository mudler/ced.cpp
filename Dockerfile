# ced.cpp container image.
#
# Multi-stage build: a fat build stage compiles ced-cli (and the ggml backends
# it links against), then a slim runtime stage carries only the binary plus the
# ggml shared libraries.
#
# The same Dockerfile produces the CPU and CUDA variants. Select with build
# args:
#
#   CPU (default):
#     docker build -t ced.cpp:cpu .
#
#   CUDA (GGML_CUDA_NO_VMM=ON drops the libcuda driver-lib link dependency,
#   which a GPU-less build container does not have):
#     docker build -t ced.cpp:cuda \
#       --build-arg BUILD_BASE=nvidia/cuda:13.0.1-devel-ubuntu24.04 \
#       --build-arg RUNTIME_BASE=nvidia/cuda:13.0.1-runtime-ubuntu24.04 \
#       --build-arg "CMAKE_EXTRA_ARGS=-DCED_GGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON" .
#
# The build context must be a checkout with the ggml submodule populated
# (git clone --recursive, or actions/checkout with submodules: recursive).
# Models are not bundled: mount a pre-converted .gguf at runtime.

ARG BUILD_BASE=ubuntu:24.04
ARG RUNTIME_BASE=ubuntu:24.04

# ---------------------------------------------------------------------------
# build: configure + compile ced-cli and the ggml backends.
# ---------------------------------------------------------------------------
FROM ${BUILD_BASE} AS build

# Extra cmake flags appended verbatim (e.g. -DCED_GGML_CUDA=ON).
ARG CMAKE_EXTRA_ARGS=""
# CUDA architectures, as a quoted CMAKE_CUDA_ARCHITECTURES list. Empty = let
# ggml pick its default broad list.
ARG CUDA_ARCHS=""

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# GGML_NATIVE=OFF keeps the binary portable across the CPUs that will pull the
# published image (no host-specific ISA extensions baked in). GGML_LLAMAFILE
# stays on (forced by CMakeLists) for the tinyBLAS SGEMM speedup.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DCED_BUILD_CLI=ON \
        -DCED_BUILD_TESTS=OFF \
        ${CMAKE_EXTRA_ARGS} \
        ${CUDA_ARCHS:+"-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"} \
    && cmake --build build -j"$(nproc)"

# Stage the binary and every backend shared library (CPU, and CUDA when built)
# into a clean prefix the runtime stage can copy wholesale.
RUN mkdir -p /install/bin /install/lib \
    && cp build/examples/cli/ced-cli /install/bin/ \
    && find build -name '*.so*' -exec cp -av {} /install/lib/ \;

# ---------------------------------------------------------------------------
# runtime: slim image with just the binary and its shared libraries.
# ---------------------------------------------------------------------------
FROM ${RUNTIME_BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /install/bin/ /usr/local/bin/
COPY --from=build /install/lib/ /usr/local/lib/
RUN ldconfig

WORKDIR /work
ENTRYPOINT ["ced-cli"]
CMD ["--help"]
