#!/usr/bin/env bash
set -euo pipefail

D2_ROOT="${D2_ROOT:-/root/autodl-tmp/D_squared_method}"
LMMS_EVAL_VERSION="${LMMS_EVAL_VERSION:-0.1.0}"
LMMS_EVAL_TAG="${LMMS_EVAL_TAG:-v${LMMS_EVAL_VERSION}}"
LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-/root/autodl-tmp/lmms-eval-${LMMS_EVAL_VERSION}}"
GITHUB_PROXY_PREFIX="${GITHUB_PROXY_PREFIX:-https://ghfast.top/}"
LMMS_EVAL_REPO="${LMMS_EVAL_REPO:-${GITHUB_PROXY_PREFIX}https://github.com/EvolvingLMMs-Lab/lmms-eval.git}"
LMMS_EVAL_REPO_FALLBACK="${LMMS_EVAL_REPO_FALLBACK:-https://github.com/EvolvingLMMs-Lab/lmms-eval.git}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-$D2_ROOT/autodl_lmms/constraints_d2_llava_lmms.txt}"

export PIP_INDEX_URL
export HF_ENDPOINT
export PYTHONPATH="$D2_ROOT/src:$D2_ROOT/src/LLaVA:$D2_ROOT/src/transformers/src:${PYTHONPATH:-}"

cleanup_failed_clone_dir() {
  if [ ! -e "$LMMS_EVAL_DIR" ]; then
    return 0
  fi
  case "$LMMS_EVAL_DIR" in
    /root/autodl-tmp/lmms-eval|/root/autodl-tmp/lmms-eval/|/root/autodl-tmp/lmms-eval-*)
      rm -rf "$LMMS_EVAL_DIR"
      ;;
    *)
      echo "Clone failed and cleanup was skipped for custom LMMS_EVAL_DIR: $LMMS_EVAL_DIR" >&2
      echo "Remove that directory manually, or set LMMS_EVAL_DIR=/root/autodl-tmp/lmms-eval." >&2
      exit 1
      ;;
  esac
}

if [ ! -d "$D2_ROOT" ]; then
  echo "D2_ROOT does not exist: $D2_ROOT" >&2
  exit 1
fi

if [ ! -f "$CONSTRAINTS_FILE" ]; then
  echo "CONSTRAINTS_FILE does not exist: $CONSTRAINTS_FILE" >&2
  exit 1
fi

if [ -d "$LMMS_EVAL_DIR" ] && [ ! -d "$LMMS_EVAL_DIR/.git" ]; then
  if [ -z "$(find "$LMMS_EVAL_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    rmdir "$LMMS_EVAL_DIR"
  else
    echo "LMMS_EVAL_DIR exists but is not a git repo: $LMMS_EVAL_DIR" >&2
    echo "Move it away or set LMMS_EVAL_DIR to another path." >&2
    exit 1
  fi
fi

if [ -d "$LMMS_EVAL_DIR/.git" ] && [ ! -f "$LMMS_EVAL_DIR/pyproject.toml" ]; then
  echo "LMMS_EVAL_DIR looks like an incomplete clone: $LMMS_EVAL_DIR" >&2
  cleanup_failed_clone_dir
fi

if [ ! -d "$LMMS_EVAL_DIR/.git" ]; then
  mkdir -p "$(dirname "$LMMS_EVAL_DIR")"
  echo "Cloning lmms-eval $LMMS_EVAL_TAG from: $LMMS_EVAL_REPO"
  if ! git clone --branch "$LMMS_EVAL_TAG" --depth 1 "$LMMS_EVAL_REPO" "$LMMS_EVAL_DIR"; then
    echo "Mirror clone failed, trying fallback: $LMMS_EVAL_REPO_FALLBACK" >&2
    cleanup_failed_clone_dir
    git clone --branch "$LMMS_EVAL_TAG" --depth 1 "$LMMS_EVAL_REPO_FALLBACK" "$LMMS_EVAL_DIR"
  fi
fi

cd "$D2_ROOT"
python -m pip install --no-deps -e "$D2_ROOT/src/LLaVA"
python -m pip install --no-deps -e "$D2_ROOT/src/transformers"

cd "$LMMS_EVAL_DIR"
python -m pip install -c "$CONSTRAINTS_FILE" -e .

cd "$D2_ROOT"
python "$D2_ROOT/autodl_lmms/patch_lmms_eval_llava.py" --lmms-eval-dir "$LMMS_EVAL_DIR"
if [ -f "$LMMS_EVAL_DIR/lmms_eval/models/simple/llava.py" ]; then
  python -m py_compile "$LMMS_EVAL_DIR/lmms_eval/models/simple/llava.py"
else
  python -m py_compile "$LMMS_EVAL_DIR/lmms_eval/models/llava.py"
fi

python - <<'PY'
import lmms_eval.models  # noqa: F401
from lmms_eval.api.registry import MODEL_REGISTRY

if "llava" not in MODEL_REGISTRY:
    raise SystemExit(f"llava is not registered; registered models: {sorted(MODEL_REGISTRY)}")
print("registered models:", sorted(MODEL_REGISTRY))
PY

echo "lmms-eval is ready at $LMMS_EVAL_DIR"
