#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/majie/code_junle}"
DQECG_ROOT="${DQECG_ROOT:-$CODE_ROOT/D-QECG}"
LMMS_EVAL_VERSION="${LMMS_EVAL_VERSION:-0.7.1}"
DQECG_ENV_PREFIX="${DQECG_ENV_PREFIX:-${CONDA_PREFIX:-/home/majie/.conda/envs/D-QECG}}"
LMMS_SOURCE_DIR="${LMMS_SOURCE_DIR:-$DQECG_ENV_PREFIX/src/lmms-eval-${LMMS_EVAL_VERSION}-mirror}"
GITHUB_ARCHIVE="https://github.com/EvolvingLMMs-Lab/lmms-eval/archive/refs/tags/v${LMMS_EVAL_VERSION}.tar.gz"

if [ ! -d "$DQECG_ROOT" ]; then
  echo "D-QECG project directory does not exist: $DQECG_ROOT" >&2
  exit 1
fi
if [ ! -f "$DQECG_ROOT/server/patch_lmms_eval_llava.py" ]; then
  echo "Missing patch script: $DQECG_ROOT/server/patch_lmms_eval_llava.py" >&2
  exit 1
fi

export PYTHONPATH="$DQECG_ROOT/src:$DQECG_ROOT/src/LLaVA:${PYTHONPATH:-}"

download_archive() {
  local url="$1"
  local output="$2"
  echo "Trying: $url"
  if command -v curl >/dev/null 2>&1; then
    curl \
      --fail \
      --location \
      --connect-timeout 20 \
      --retry 3 \
      --retry-all-errors \
      --output "$output" \
      "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget \
      --timeout=20 \
      --tries=3 \
      --output-document="$output" \
      "$url"
  else
    echo "Neither curl nor wget is available." >&2
    return 1
  fi
}

validate_source() {
  local source="$1"
  local required=(
    "pyproject.toml"
    "lmms_eval/models/model_utils/usage_metrics.py"
    "lmms_eval/models/simple/llava.py"
    "lmms_eval/tasks/mme/mme.yaml"
    "lmms_eval/tasks/embspatial/_default_template_yaml"
  )
  local item
  for item in "${required[@]}"; do
    if [ ! -e "$source/$item" ]; then
      echo "Incomplete lmms-eval archive; missing: $item" >&2
      return 1
    fi
  done
}

ensure_model_utils_package() {
  local package_dir="$LMMS_SOURCE_DIR/lmms_eval/models/model_utils"
  local package_init="$package_dir/__init__.py"

  if [ ! -f "$package_dir/usage_metrics.py" ]; then
    echo "Missing lmms-eval usage metrics module: $package_dir/usage_metrics.py" >&2
    return 1
  fi

  # v0.7.1's source archive can contain model_utils without an __init__.py.
  # setuptools then omits the directory from an editable installation, although
  # llm_judge imports it unconditionally while starting every evaluation.
  if [ ! -f "$package_init" ]; then
    printf '%s\n' \
      '"""Model utility helpers (package marker added by D-QECG)."""' \
      >"$package_init"
    echo "Added missing package marker: $package_init"
  fi
}

if validate_source "$LMMS_SOURCE_DIR" >/dev/null 2>&1; then
  echo "Reusing complete lmms-eval source: $LMMS_SOURCE_DIR"
else
  if [ -e "$LMMS_SOURCE_DIR" ]; then
    echo "Target source path exists but is incomplete: $LMMS_SOURCE_DIR" >&2
    echo "Move that directory away, then run this script again." >&2
    exit 1
  fi

  DOWNLOAD_DIR="$(mktemp -d)"
  trap 'rm -rf -- "$DOWNLOAD_DIR"' EXIT
  ARCHIVE_PATH="$DOWNLOAD_DIR/lmms-eval-${LMMS_EVAL_VERSION}.tar.gz"
  MIRROR_URLS=(
    "https://gh-fast.com/$GITHUB_ARCHIVE"
    "https://gh-proxy.com/$GITHUB_ARCHIVE"
    "https://ghproxy.net/$GITHUB_ARCHIVE"
    "https://codeload.github.com/EvolvingLMMs-Lab/lmms-eval/tar.gz/refs/tags/v${LMMS_EVAL_VERSION}"
  )

  DOWNLOAD_OK=false
  for url in "${MIRROR_URLS[@]}"; do
    if download_archive "$url" "$ARCHIVE_PATH" \
      && tar -tzf "$ARCHIVE_PATH" >/dev/null 2>&1; then
      DOWNLOAD_OK=true
      echo "Downloaded a valid archive from: $url"
      break
    fi
    echo "Mirror failed or returned an invalid archive." >&2
  done
  if [ "$DOWNLOAD_OK" != "true" ]; then
    echo "All GitHub archive mirrors failed." >&2
    exit 1
  fi

  EXTRACT_DIR="$DOWNLOAD_DIR/extracted"
  mkdir -p "$EXTRACT_DIR"
  tar -xzf "$ARCHIVE_PATH" -C "$EXTRACT_DIR"
  EXTRACTED_SOURCE="$(
    find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d -print -quit
  )"
  if [ -z "$EXTRACTED_SOURCE" ]; then
    echo "Downloaded archive has an unexpected layout." >&2
    exit 1
  fi
  validate_source "$EXTRACTED_SOURCE"

  mkdir -p "$(dirname "$LMMS_SOURCE_DIR")"
  mv "$EXTRACTED_SOURCE" "$LMMS_SOURCE_DIR"
fi

ensure_model_utils_package

# Prefer the complete checkout over setuptools' editable package map. Some
# v0.7.1 builds omit model_utils from that map even though the source exists.
export PYTHONPATH="$LMMS_SOURCE_DIR:$PYTHONPATH"

echo "Installing complete lmms-eval source without changing existing dependencies."
python -m pip install \
  --no-deps \
  --no-build-isolation \
  -e "$LMMS_SOURCE_DIR"

python "$DQECG_ROOT/server/patch_lmms_eval_llava.py" \
  --lmms-eval-dir "$LMMS_SOURCE_DIR"

python -m py_compile \
  "$LMMS_SOURCE_DIR/lmms_eval/models/simple/llava.py"

python - <<'PY'
import importlib.metadata
from lmms_eval import evaluator
from lmms_eval.models.model_utils.usage_metrics import log_usage
from lmms_eval.tasks import TaskManager

assert evaluator is not None
assert callable(log_usage)
TaskManager()
print("lmms-eval:", importlib.metadata.version("lmms-eval"))
print("Evaluator import: ok")
print("Task template scan: ok")
PY

echo "lmms-eval mirror runtime is ready: $LMMS_SOURCE_DIR"
