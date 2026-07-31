#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

CODE_ROOT="${CODE_ROOT:-/home/majie/code_junle}"
DATA_ROOT="${DATA_ROOT:-/home/majie/majie_data}"
DQECG_ROOT="${DQECG_ROOT:-$CODE_ROOT/D-QECG}"
model_path="${MODEL_PATH:-$DATA_ROOT/base_model/llava-v1.5-13b}"
output_path="${OUTPUT_PATH:-$DATA_ROOT/D-QECG/aokvqa_eval}"
export DATA_ROOT
export PYTHONPATH="$DQECG_ROOT/src:$DQECG_ROOT/src/LLaVA:${PYTHONPATH:-}"
mkdir -p "$output_path"

rank_list=(72 144 288 432)
Ks=(2)

for rank in "${rank_list[@]}"; do
    for k in "${Ks[@]}"; do
        python "$DQECG_ROOT/src/dqecg/inference/eval/inference_aokvqa.py" \
            --model-path "$model_path" \
            --use-dqecg \
            --d-squared-inplace \
            --d-squared-sys-length 36 \
            --d-squared-image-token-length 576 \
            --d-squared-visual-keep-count "$rank" \
            --d-squared-llm-keep-count "$rank" \
            --d-squared-agg-layer "$k" \
            --output-path "$output_path/aokvqa_13b_d_squared_4bit_inplace_${rank}_${k}.json"
    done
done

python "$DQECG_ROOT/src/dqecg/inference/eval/inference_aokvqa.py" \
    --model-path "$model_path" \
    --output-path "$output_path/aokvqa_13b_baseline.json"
