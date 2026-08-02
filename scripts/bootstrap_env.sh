#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-uv}"
PYTHON_BIN="${EMBER_BASE_PYTHON:-/usr/bin/python3}"
VENV_DIR="${UV_PROJECT_ENVIRONMENT:-${ROOT}/.venv}"
: "${EMBER_CACHE_ROOT:?Set EMBER_CACHE_ROOT to a storage-backed cache directory}"

export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export UV_CACHE_DIR="${EMBER_CACHE_ROOT}/uv"
export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"

cd "${ROOT}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv --python "${PYTHON_BIN}" "${VENV_DIR}"
fi
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" cmake==3.31.6 ziglang==0.16.0

export PATH="${VENV_DIR}/bin:${PATH}"
export EMBER_PYTHON="${VENV_DIR}/bin/python"
export CXX="${ROOT}/scripts/zig-cxx"
"${UV_BIN}" sync --locked
"${EMBER_PYTHON}" -m ember.runtime_env
"${UV_BIN}" pip check --python "${EMBER_PYTHON}"
"${UV_BIN}" sync --locked --check
