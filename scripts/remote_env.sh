#!/usr/bin/env bash
set -euo pipefail

export SPECSTREAM_ROOT=/root/dataDisk/specstream
export HF_HOME="${SPECSTREAM_ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="${SPECSTREAM_ROOT}/cache/datasets"
export TRANSFORMERS_CACHE="${SPECSTREAM_ROOT}/cache/transformers"
export TMPDIR="${SPECSTREAM_ROOT}/cache/tmp"
export PYTHONUNBUFFERED=1

mkdir -p \
  "${SPECSTREAM_ROOT}/artifacts" \
  "${SPECSTREAM_ROOT}/cache/datasets" \
  "${SPECSTREAM_ROOT}/cache/tmp" \
  "${SPECSTREAM_ROOT}/cache/transformers" \
  "${SPECSTREAM_ROOT}/logs"

source "${SPECSTREAM_ROOT}/.venv/bin/activate"

