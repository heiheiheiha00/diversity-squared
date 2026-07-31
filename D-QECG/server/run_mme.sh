#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/majie/code_junle}"
DATA_ROOT="${DATA_ROOT:-/home/majie/majie_data}"
DQECG_ROOT="${DQECG_ROOT:-$CODE_ROOT/D-QECG}"
MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/base_model/llava-v1.5-7b}"
OUTPUT_PATH="${OUTPUT_PATH:-$DQECG_ROOT/outputs/lmms_eval_logs}"
MODE="${MODE:-baseline}"
TASKS="${TASKS:-mme}"
LMMS_SOURCE_DIR="${LMMS_SOURCE_DIR:-${CONDA_PREFIX:-/home/majie/.conda/envs/D-QECG}/src/lmms-eval-0.7.1-mirror}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29507}"
BATCH_SIZE="${BATCH_SIZE:-1}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
if [ -z "${NUM_PROCESSES:-}" ]; then
  IFS=',' read -r -a VISIBLE_GPU_LIST <<<"$CUDA_VISIBLE_DEVICES"
  NUM_PROCESSES="${#VISIBLE_GPU_LIST[@]}"
fi
export HF_HOME="${HF_HOME:-/home/majie/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-$TRANSFORMERS_OFFLINE}"
if [ -d "$LMMS_SOURCE_DIR/lmms_eval" ]; then
  export PYTHONPATH="$LMMS_SOURCE_DIR:$DQECG_ROOT/src:$DQECG_ROOT/src/LLaVA:${PYTHONPATH:-}"
else
  export PYTHONPATH="$DQECG_ROOT/src:$DQECG_ROOT/src/LLaVA:${PYTHONPATH:-}"
fi
MME_CACHE_DIR="${MME_CACHE_DIR:-$HF_DATASETS_CACHE/lmms-lab___mme}"

if [ ! -d "$DQECG_ROOT" ]; then
  echo "DQECG_ROOT does not exist: $DQECG_ROOT" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c "from lmms_eval import evaluator" >/dev/null 2>&1; then
  echo "The complete lmms-eval evaluator is unavailable in the active environment." >&2
  echo "Run server/install_lmms_eval_only.sh first." >&2
  exit 1
fi
if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "LLaVA model config does not exist: $MODEL_PATH/config.json" >&2
  exit 1
fi
if [[ ",$TASKS," == *",mme,"* ]] \
  && [ "$HF_DATASETS_OFFLINE" = "1" ] \
  && [ ! -d "$MME_CACHE_DIR" ]; then
  echo "Offline MME cache does not exist: $MME_CACHE_DIR" >&2
  exit 1
fi
if [ -n "${LIMIT:-}" ] && (( LIMIT % 2 != 0 )); then
  echo "LIMIT must be even for paired MME metrics; got $LIMIT." >&2
  exit 1
fi
if ! [[ "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_PROCESSES must be a positive integer; got $NUM_PROCESSES." >&2
  exit 1
fi
if ! [[ "$MAIN_PROCESS_PORT" =~ ^[0-9]+$ ]] \
  || (( MAIN_PROCESS_PORT < 1 || MAIN_PROCESS_PORT > 65535 )); then
  echo "MAIN_PROCESS_PORT must be between 1 and 65535; got $MAIN_PROCESS_PORT." >&2
  exit 1
fi

BASE_ARGS="pretrained=$MODEL_PATH,conv_template=vicuna_v1"
case "$MODE" in
  baseline)
    ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
    MODEL_ARGS="$BASE_ARGS,attn_implementation=$ATTN_IMPLEMENTATION"
    SUFFIX="${LOG_SUFFIX:-llava_v1_5_7b_mme_baseline}"
    ;;
  dqecg|d_squared)
    D2_SYS_LENGTH="${D2_SYS_LENGTH:-36}"
    D2_IMAGE_TOKENS="${D2_IMAGE_TOKENS:-576}"
    D2_VISUAL_KEEP="${D2_VISUAL_KEEP:-32}"
    D2_LLM_KEEP="${D2_LLM_KEEP:-32}"
    D2_AGG_LAYER="${D2_AGG_LAYER:-3}"
    D2_QCEG_TAU="${D2_QCEG_TAU:-0.1}"
    D2_SKIP_VISUAL_SELECTOR="${D2_SKIP_VISUAL_SELECTOR:-false}"
    D2_USE_CACHE="${D2_USE_CACHE:-true}"
    D2_STATIC_KV_CACHE="${D2_STATIC_KV_CACHE:-$D2_USE_CACHE}"
    D2_ATTN_IMPLEMENTATION="${D2_ATTN_IMPLEMENTATION:-eager}"
    MODEL_ARGS="$BASE_ARGS,attn_implementation=$D2_ATTN_IMPLEMENTATION,use_cache=$D2_USE_CACHE,use_d_squared=true,d_squared_inplace=true,d_squared_static_kv_cache=$D2_STATIC_KV_CACHE,d_squared_skip_visual_selector=$D2_SKIP_VISUAL_SELECTOR,d_squared_sys_length=$D2_SYS_LENGTH,d_squared_image_token_length=$D2_IMAGE_TOKENS,d_squared_visual_keep_count=$D2_VISUAL_KEEP,d_squared_llm_keep_count=$D2_LLM_KEEP,d_squared_agg_layer=$D2_AGG_LAYER,d_squared_second_stage_method=qceg,d_squared_qceg_tau=$D2_QCEG_TAU,d_squared_query_score_mode=question_only,d_squared_dump_selection_debug=false"
    SUFFIX="${LOG_SUFFIX:-llava_v1_5_7b_mme_dqecg_v${D2_VISUAL_KEEP}_l${D2_LLM_KEEP}}"
    ;;
  *)
    echo "Unsupported MODE: $MODE. Use baseline or dqecg." >&2
    exit 1
    ;;
esac

EXTRA_ARGS=()
if [ -n "${LIMIT:-}" ]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi

mkdir -p "$OUTPUT_PATH"
cd "$DQECG_ROOT"
echo "Using DQECG_ROOT=$DQECG_ROOT"
echo "Using MODEL_PATH=$MODEL_PATH"
echo "Using MME_CACHE_DIR=$MME_CACHE_DIR"
echo "Using OUTPUT_PATH=$OUTPUT_PATH"
echo "Using model_args=$MODEL_ARGS"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using NUM_PROCESSES=$NUM_PROCESSES"
echo "Using MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"

"$PYTHON_BIN" -m accelerate.commands.launch \
  --main_process_port="$MAIN_PROCESS_PORT" \
  --num_processes="$NUM_PROCESSES" \
  --num_machines=1 \
  --mixed_precision="${MIXED_PRECISION:-no}" \
  --dynamo_backend=no \
  -m lmms_eval \
  --model llava \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --log_samples \
  --log_samples_suffix "$SUFFIX" \
  --output_path "$OUTPUT_PATH" \
  "${EXTRA_ARGS[@]}"
