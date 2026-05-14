export CUDA_VISIBLE_DEVICES=0

model_path=./models/llava-v1.5-7b
output_path=ocrvqa_eval_d_squared
mkdir -p $output_path

rank_list=(72 144 288 432)
Ks=(2)

for rank in ${rank_list[@]}; do
    for k in ${Ks[@]}; do
    python ./src/D_squared_method/inference/eval/inference_ocrvqa.py \
        --model-path $model_path \
        --use-d-squared \
        --d-squared-sys-length 36 \
        --d-squared-image-token-length 576 \
        --d-squared-visual-keep-count $rank \
        --d-squared-llm-keep-count $rank \
        --d-squared-agg-layer $k \
        --output-path $output_path/ocrvqa_7b_d_squared_nocache_mt40_${rank}_${k}.json
    done
done
