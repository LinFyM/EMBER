#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-uv}"
: "${EMBER_CACHE_ROOT:?Set EMBER_CACHE_ROOT to a storage-backed cache directory}"

export UV_CACHE_DIR="${EMBER_CACHE_ROOT}/uv"
export UV_LINK_MODE=hardlink

cd "${ROOT}"
"${UV_BIN}" venv --python /usr/bin/python3 --allow-existing .venv
"${UV_BIN}" pip install --python .venv/bin/python cmake==3.31.6 ziglang==0.16.0

export PATH="${ROOT}/.venv/bin:${PATH}"
export EMBER_PYTHON="${ROOT}/.venv/bin/python"
export CXX="${ROOT}/scripts/zig-cxx"
"${UV_BIN}" sync --locked
"${EMBER_PYTHON}" -m ember.runtime_env
"${UV_BIN}" pip check --python "${EMBER_PYTHON}"
"${UV_BIN}" sync --locked --check
