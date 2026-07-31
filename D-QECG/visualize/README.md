# D-QECG 可视化工具

脚本保存在项目内，生成的 dump 和图片写到数据目录。

服务器运行：

```bash
cd /home/majie/code_junle/D-QECG
export PYTHONPATH="$PWD/src:$PWD/src/LLaVA:${PYTHONPATH:-}"

python visualize/run_selection_debug_dump.py \
  --model-path /home/majie/majie_data/base_model/llava-v1.5-7b \
  --cases /home/majie/majie_data/sink_residual_20cases/cases.jsonl \
  --output-dir /home/majie/majie_data/D-QECG/visualize/qceg_debug_k64 \
  --visual-token-count 576 \
  --visual-keep-count 32 \
  --llm-keep-count 32 \
  --d-squared-agg-layer 3 \
  --second-stage-method qceg \
  --qceg-tau 0.1 \
  --query-score-mode question_only \
  --question-source auto \
  --d-squared-inplace
```

本地绘图：

```bash
python D-QECG/visualize/local_selection_viz.py \
  --dump-dir E:\new_version\qceg_debug_k64 \
  --save-dir E:\new_version\visualize\figures\qceg_debug_k64 \
  --image-root E:\new_version\sink_residual_20cases
```
