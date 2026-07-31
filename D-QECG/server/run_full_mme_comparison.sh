#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/majie/code_junle}"
DQECG_ROOT="${DQECG_ROOT:-$CODE_ROOT/D-QECG}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REPORT_ROOT="${REPORT_ROOT:-$DQECG_ROOT/outputs/mme_full_comparison/$RUN_ID}"
MODEL_PATH="${MODEL_PATH:-/home/majie/majie_data/base_model/llava-v1.5-7b}"
MME_CACHE_DIR="${MME_CACHE_DIR:-/home/majie/.cache/huggingface/datasets/lmms-lab___mme}"
ATTN_BACKEND="${ATTN_BACKEND:-eager}"
LIMIT="${LIMIT:-}"
LMMS_SOURCE_DIR="${LMMS_SOURCE_DIR:-${CONDA_PREFIX:-/home/majie/.conda/envs/D-QECG}/src/lmms-eval-0.7.1-mirror}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -d "$LMMS_SOURCE_DIR/lmms_eval" ]; then
  export PYTHONPATH="$LMMS_SOURCE_DIR:$DQECG_ROOT/src:$DQECG_ROOT/src/LLaVA:${PYTHONPATH:-}"
else
  export PYTHONPATH="$DQECG_ROOT/src:$DQECG_ROOT/src/LLaVA:${PYTHONPATH:-}"
fi

if [ ! -f "$DQECG_ROOT/server/run_mme.sh" ]; then
  echo "Missing evaluator: $DQECG_ROOT/server/run_mme.sh" >&2
  exit 1
fi
if [ ! -f "$DQECG_ROOT/server/summarize_mme_scores.py" ]; then
  echo "Missing summarizer: $DQECG_ROOT/server/summarize_mme_scores.py" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c "from lmms_eval import evaluator" >/dev/null 2>&1; then
  echo "The complete lmms-eval evaluator is unavailable in the active environment." >&2
  echo "Run server/install_lmms_eval_only.sh first." >&2
  exit 1
fi
if [ ! -d "$MME_CACHE_DIR" ]; then
  echo "MME cache does not exist: $MME_CACHE_DIR" >&2
  exit 1
fi
if [ -n "$LIMIT" ] && (( LIMIT % 2 != 0 )); then
  echo "LIMIT must be even for paired MME metrics; got $LIMIT." >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-/home/majie/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MODEL_PATH MME_CACHE_DIR

mkdir -p "$REPORT_ROOT/baseline" "$REPORT_ROOT/dqecg"

require_result_json() {
  local result_dir="$1"
  "$PYTHON_BIN" - "$result_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in root.rglob("*.json"):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
        print(f"Validated result: {path}")
        raise SystemExit(0)
raise SystemExit(f"lmms-eval produced no result JSON under {root}")
PY
}

echo "=== MME baseline: no pruning, attention=$ATTN_BACKEND ==="
OUTPUT_PATH="$REPORT_ROOT/baseline" \
LOG_SUFFIX=mme_baseline \
MODE=baseline \
ATTN_IMPLEMENTATION="$ATTN_BACKEND" \
LIMIT="$LIMIT" \
bash "$DQECG_ROOT/server/run_mme.sh" \
  2>&1 | tee "$REPORT_ROOT/baseline.log"
require_result_json "$REPORT_ROOT/baseline"

echo "=== MME D-QECG: pruning enabled, attention=$ATTN_BACKEND ==="
OUTPUT_PATH="$REPORT_ROOT/dqecg" \
LOG_SUFFIX=mme_dqecg \
MODE=dqecg \
D2_USE_CACHE=true \
D2_STATIC_KV_CACHE=true \
D2_ATTN_IMPLEMENTATION="$ATTN_BACKEND" \
LIMIT="$LIMIT" \
bash "$DQECG_ROOT/server/run_mme.sh" \
  2>&1 | tee "$REPORT_ROOT/dqecg.log"
require_result_json "$REPORT_ROOT/dqecg"

"$PYTHON_BIN" "$DQECG_ROOT/server/summarize_mme_scores.py" \
  --baseline-dir "$REPORT_ROOT/baseline" \
  --dqecg-dir "$REPORT_ROOT/dqecg" \
  --output-dir "$REPORT_ROOT"

echo "MME comparison completed: $REPORT_ROOT"
