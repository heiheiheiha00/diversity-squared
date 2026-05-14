export CUDA_VISIBLE_DEVICES=0

model_path=/cpfs01/user/cl424408/models/llava-v1.5-13b
output_path=aokvqa_eval_d_squared
mkdir -p $output_path

rank_list=(72 144 288 432)
Ks=(2)

for rank in ${rank_list[@]}; do
    for k in ${Ks[@]}; do
    python ./src/D_squared_method/inference/eval/inference_aokvqa.py \
        --model-path $model_path \
        --use-d-squared \
        --d-squared-inplace \
        --d-squared-sys-length 36 \
        --d-squared-image-token-length 576 \
        --d-squared-visual-keep-count $rank \
        --d-squared-llm-keep-count $rank \
        --d-squared-agg-layer $k \
        --output-path $output_path/aokvqa_13b_d_squared_4bit_inplace_${rank}_${k}.json
    done
done

python ./src/D_squared_method/inference/eval/inference_aokvqa.py \
    --model-path $model_path \
    --output-path $output_path/aokvqa_13b_baseline.json
