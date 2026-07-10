#!/usr/bin/env bash
set -euo pipefail

D2_ROOT="${D2_ROOT:-/root/autodl-tmp/D_squared_method}"
LMMS_EVAL_VERSION="${LMMS_EVAL_VERSION:-0.1.0}"
LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-/root/autodl-tmp/lmms-eval-${LMMS_EVAL_VERSION}}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/llava-v1.5-7b}"
OUTPUT_PATH="${OUTPUT_PATH:-$LMMS_EVAL_DIR/logs}"
MODE="${MODE:-baseline}"
TASKS="${TASKS:-mme}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTHONPATH="$D2_ROOT/src:$D2_ROOT/src/LLaVA:$D2_ROOT/src/transformers/src:${PYTHONPATH:-}"

if [ ! -d "$LMMS_EVAL_DIR" ]; then
  echo "LMMS_EVAL_DIR does not exist: $LMMS_EVAL_DIR" >&2
  echo "Run autodl_lmms/setup_autodl.sh first." >&2
  exit 1
fi

if [ ! -e "$MODEL_PATH" ]; then
  echo "MODEL_PATH does not exist: $MODEL_PATH" >&2
  echo "Set MODEL_PATH to the real LLaVA checkpoint directory under /root/autodl-tmp/models." >&2
  if [ -d "/root/autodl-tmp/models" ]; then
    echo "Available entries in /root/autodl-tmp/models:" >&2
    find /root/autodl-tmp/models -maxdepth 2 -mindepth 1 -type d | sort >&2
  fi
  exit 1
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "Warning: $MODEL_PATH/config.json was not found." >&2
  echo "Make sure MODEL_PATH points to the actual Hugging Face/LLaVA checkpoint directory, not only the parent models folder." >&2
fi

if find "$MODEL_PATH" -maxdepth 1 -type f \( -name "pytorch_model*.bin" -o -name "*.safetensors" \) -size 0c | grep -q .; then
  echo "Warning: some model weight files under $MODEL_PATH are 0 bytes." >&2
  echo "The directory/config is visible, but checkpoint upload is not complete yet." >&2
  find "$MODEL_PATH" -maxdepth 1 -type f \( -name "pytorch_model*.bin" -o -name "*.safetensors" \) -size 0c -print >&2
fi

BASE_ARGS="pretrained=$MODEL_PATH,conv_template=vicuna_v1"
case "$MODE" in
  baseline)
    MODEL_ARGS="$BASE_ARGS"
    SUFFIX="${LOG_SUFFIX:-llava_v1_5_7b_mme_official}"
    ;;
  d_squared)
    D2_SYS_LENGTH="${D2_SYS_LENGTH:-36}"
    D2_IMAGE_TOKENS="${D2_IMAGE_TOKENS:-576}"
    D2_VISUAL_KEEP="${D2_VISUAL_KEEP:-32}"
    D2_LLM_KEEP="${D2_LLM_KEEP:-32}"
    D2_AGG_LAYER="${D2_AGG_LAYER:-3}"
    D2_QCEG_TAU="${D2_QCEG_TAU:-0.1}"
    D2_SKIP_VISUAL_SELECTOR="${D2_SKIP_VISUAL_SELECTOR:-false}"
    D2_USE_CACHE="${D2_USE_CACHE:-true}"
    D2_STATIC_KV_CACHE="${D2_STATIC_KV_CACHE:-$D2_USE_CACHE}"
    MODEL_ARGS="$BASE_ARGS,use_cache=$D2_USE_CACHE,use_d_squared=true,d_squared_inplace=true,d_squared_static_kv_cache=$D2_STATIC_KV_CACHE,d_squared_skip_visual_selector=$D2_SKIP_VISUAL_SELECTOR,d_squared_sys_length=$D2_SYS_LENGTH,d_squared_image_token_length=$D2_IMAGE_TOKENS,d_squared_visual_keep_count=$D2_VISUAL_KEEP,d_squared_llm_keep_count=$D2_LLM_KEEP,d_squared_agg_layer=$D2_AGG_LAYER,d_squared_second_stage_method=qceg,d_squared_qceg_tau=$D2_QCEG_TAU,d_squared_query_score_mode=question_only,d_squared_dump_selection_debug=false"
    SUFFIX="${LOG_SUFFIX:-llava_v1_5_7b_mme_d_squared_qceg_v${D2_VISUAL_KEEP}_l${D2_LLM_KEEP}}"
    ;;
  llm_selector|llm_only)
    D2_SYS_LENGTH="${D2_SYS_LENGTH:-36}"
    D2_IMAGE_TOKENS="${D2_IMAGE_TOKENS:-576}"
    D2_VISUAL_KEEP="${D2_VISUAL_KEEP:-0}"
    D2_LLM_KEEP="${D2_LLM_KEEP:-64}"
    D2_AGG_LAYER="${D2_AGG_LAYER:-3}"
    D2_QCEG_TAU="${D2_QCEG_TAU:-0.1}"
    D2_SKIP_VISUAL_SELECTOR="${D2_SKIP_VISUAL_SELECTOR:-true}"
    D2_USE_CACHE="${D2_USE_CACHE:-true}"
    D2_STATIC_KV_CACHE="${D2_STATIC_KV_CACHE:-$D2_USE_CACHE}"
    MODEL_ARGS="$BASE_ARGS,use_cache=$D2_USE_CACHE,use_d_squared=true,d_squared_inplace=true,d_squared_static_kv_cache=$D2_STATIC_KV_CACHE,d_squared_skip_visual_selector=$D2_SKIP_VISUAL_SELECTOR,d_squared_sys_length=$D2_SYS_LENGTH,d_squared_image_token_length=$D2_IMAGE_TOKENS,d_squared_visual_keep_count=$D2_VISUAL_KEEP,d_squared_llm_keep_count=$D2_LLM_KEEP,d_squared_agg_layer=$D2_AGG_LAYER,d_squared_second_stage_method=qceg,d_squared_qceg_tau=$D2_QCEG_TAU,d_squared_query_score_mode=question_only,d_squared_dump_selection_debug=false"
    SUFFIX="${LOG_SUFFIX:-llava_v1_5_7b_mme_llm_selector_l${D2_LLM_KEEP}}"
    ;;
  *)
    echo "Unsupported MODE: $MODE. Use baseline, d_squared, or llm_selector." >&2
    exit 1
    ;;
esac

EXTRA_ARGS=()
if [ -n "${LIMIT:-}" ]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi

cd "$LMMS_EVAL_DIR"
echo "Using MODEL_PATH=$MODEL_PATH"
echo "Using model_args=$MODEL_ARGS"
accelerate launch \
  --num_processes="${NUM_PROCESSES:-1}" \
  --num_machines=1 \
  --mixed_precision="${MIXED_PRECISION:-no}" \
  --dynamo_backend=no \
  -m lmms_eval \
  --model llava \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix "$SUFFIX" \
  --output_path "$OUTPUT_PATH" \
  "${EXTRA_ARGS[@]}"
