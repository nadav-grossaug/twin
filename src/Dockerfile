# ### 
# ### How to run with Podman Docker on EC2
# ###
# screen -S jupyter
#
# podman build -t ml_training:slim
# cd /home/ec2-user/twin_basedir/user_an/git/twin/src
# podman build -t twin_docker:1.0.0 .
#
# cd /home/ec2-user/twin_basedir/user_an/git/twin/src
# podman run \
# -p 60070:60070 -p 6006:6006 \
# -p 8600-8699:8600-8699 \
# -v /home/ec2-user/twin_basedir:/home/ds/twin \
# -v ~/.ssh/id_rsa.pub:/root/.ssh/id_rsa.pub:ro \
# -v ~/.ssh/id_rsa:/root/.ssh/id_rsa:ro \
# -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
# -it twin_docker:1.0.0  /bin/bash

# # run inside the docker inside screen 
#     # export PATH=/opt/venv/bin:$PATH
# cd /home/ds
# pip install 'marimo[sandbox]>=0.25.0' marimo-jupyter-extension
# jupyter lab \
# --NotebookApp.allow_remote_access=True \
# --NotebookApp.port=60070 \
# --NotebookApp.ip='0.0.0.0' \
# --NotebookApp.allow_origin='*' \
# --NotebookApp.open_browser=False \
# --NotebookApp.password='sha1:749ee7f6fa76:c6efd15c0ea3ada41f83952407354302c11cc254' \
# --allow-root --notebook-dir  /home/ds/ -y


# # open ip-172-31-26-135.us-west-2.compute.internal
# http://ip-172-31-26-135.us-west-2.compute.internal/tree/twin/user_an/
# http://35.93.9.54/tree/twin/user_an/git/twin
# http://ec2-35-93-9-54.us-west-2.compute.amazonaws.com/tree/twin


# # conneect to docker
# podman exec -it $(podman ps -q --filter ancestor=twin_docker | head -n 1) /bin/bash
# # -or- login into specific user:
# podman exec -it $(podman ps -q --filter ancestor=twin_docker | head -n 1) /bin/bash -C /home/ds/twin/user_an




# Enclave stock-series encoder — TSV -> dbt/DuckDB -> PyTorch -> Lean4+SampCert
#
# Covers every Makefile flow:
#   dbt/ml          dbt-duckdb + duckdb (requirements.txt)
#   train           PyTorch (CPU wheel, installed before requirements.txt)
#   lean*           elan + pinned Lean toolchain (lean/lean-toolchain); mathlib
#                   cache is fetched on the first `make lean-build` (~5 GB,
#                   lands in lean/.lake inside the mounted workspace)
#   notebook*       marimo + altair
#   test/manifest   pytest, pyyaml, git (provenance fingerprints)
#   synth/bundle    bash + the Lean toolchain above
#   agent tooling   Claude Code CLI + lean-ctx MCP engine
#
# Build (from twin/src):  docker build -t lean4-stockpredict .
# Run:    docker run -it -v "$PWD":/workspace -p 2718:2718 lean4-stockpredict \
#         (then: cd algorithms/algo_lean4/lean4_stockpredict)
#         (marimo inside the container: marimo run --host 0.0.0.0 visualize_enclave_mo.py)

# Base: Ubuntu 24.04 LTS (Noble) — ships Python 3.12, which the requirements
# pins target (course MacBook environment).
FROM ubuntu:24.04

ARG LEAN_CTX_VERSION=3.12.0

ENV VENV=/opt/venv
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive
ENV SPARK_LOCAL_DIRS=/opt/data/spark_tmp
ENV HDF5_USE_FILE_LOCKING=FALSE

# build-essential: SampCert FFI + native wheels; libgmp-dev: Lean/mathlib;
# libgomp1: lean-ctx onnxruntime; git/curl: lake package fetch + provenance;
# default-jdk-headless: pyspark; libhdf5-dev: h5py/tensorflow
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        build-essential libgmp-dev libgomp1 \
        default-jdk-headless libhdf5-dev openssh-client less \
        ca-certificates curl git procps htop vim screen sqlite3 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/data/spark_tmp /opt/data/spark_jars

# Python virtual environment
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# PyTorch CPU-only build (the default PyPI build bundles ~6GB of CUDA libraries);
# the pin must match requirements.txt's `torch==` so pip doesn't reinstall from PyPI
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Python dependencies: course ml_training stack (spark, tensorflow, notebooks,
# cloud, LLM/MCP) + this project's dbt-duckdb, duckdb, marimo, altair, pytest
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Lean toolchain via elan, pinned to this repo's lean/lean-toolchain so the
# toolchain is baked into the image instead of downloaded on first `lake` run
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
        | sh -s -- -y --default-toolchain none
ENV PATH="/root/.elan/bin:${PATH}"
COPY algorithms/algo_lean4/lean4_stockpredict/lean/lean-toolchain /tmp/lean-toolchain
RUN elan toolchain install "$(cat /tmp/lean-toolchain)" \
    && elan default "$(cat /tmp/lean-toolchain)"

# lean-ctx engine (single static binary from the official GitHub release)
RUN arch="$(dpkg --print-architecture)" \
    && case "$arch" in amd64) triple=x86_64 ;; arm64) triple=aarch64 ;; \
         *) echo "unsupported arch: $arch" >&2; exit 1 ;; esac \
    && curl -fsSL "https://github.com/yvgude/lean-ctx/releases/download/v${LEAN_CTX_VERSION}/lean-ctx-${triple}-unknown-linux-gnu.tar.gz" \
        | tar -xz -C /usr/local/bin lean-ctx \
    && lean-ctx --version

# Claude Code CLI (official native installer; puts the `claude` binary in /root/.local/bin)
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/root/.local/bin:${PATH}"
RUN claude --version

# marimo default port
EXPOSE 2718

WORKDIR /workspace

CMD ["bash"]
