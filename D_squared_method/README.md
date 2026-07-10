# D-squared QCEG Final

This is the cleaned final version of the D-squared code path. The repository keeps a single implementation directory named `D_squared_method`.

## Main Behavior

- Visual pruning stage is unchanged.
- LLM-side secondary visual-token selection uses QCEG by default: question-conditioned entropy gain from four early hidden states.
- Only user question tokens participate in QCEG conditioning. LLaVA wrapper tokens such as BOS/newline/`ASSISTANT:` are excluded.
- `question_only` / BOS-free / all-query attention scores are kept only as debug and ablation references.
- Formal evaluation defaults to no debug dump and no visualization output.

Default D-squared config keys:

```python
model.config.d_squared_second_stage_method = "qceg"
model.config.d_squared_qceg_tau = 0.1
model.config.d_squared_query_score_mode = "question_only"
model.config.d_squared_dump_selection_debug = False
model.config.d_squared_static_kv_cache = True
model.model.d_squared_question_query_positions = question_positions
```

## Question Span Utility

Question-token span extraction lives in:

```text
src/D_squared_method/question_span.py
```

For LLaVA-v1.5, use `build_llava_prompt_with_question_spans`, `char_spans_to_original_token_positions`, and `map_original_positions_to_expanded` to map the original question text to expanded LLM sequence positions after the image token is expanded into visual tokens.

## Debug Dump On AutoDL

Debug dump is opt-in and is not part of the clean performance path.
Use the standalone toolkit at `../visualize_toolkit` for dump and plotting scripts. Generated dumps and figures should stay outside the code directory.

```bash
cd /root/autodl-tmp
export PYTHONPATH="/root/autodl-tmp/D_squared_method/src:/root/autodl-tmp/D_squared_method/src/LLaVA:/root/autodl-tmp/D_squared_method/src/transformers/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

python visualize_toolkit/run_selection_debug_dump.py \
  --model-path /root/autodl-tmp/models/llava-v1.5-7b \
  --cases /root/autodl-tmp/sink_residual_20cases/cases.jsonl \
  --output-dir /root/autodl-tmp/visualize/outputs/qceg_debug_k64 \
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

Compress results for local visualization:

```bash
cd /root/autodl-tmp
zip -r qceg_debug_k64.zip visualize/outputs/qceg_debug_k64
```

## Local Visualization

Run locally after pulling debug `.pt` files back:

```bash
python E:\new_version\visualize_toolkit\local_selection_viz.py \
  --dump-dir E:\new_version\qceg_debug_k64\visualize\outputs\qceg_debug_k64 \
  --save-dir E:\new_version\visualize\figures\qceg_debug_k64 \
  --image-root E:\new_version\sink_residual_20cases
```

The local tool writes panels, QCEG score/gain heatmaps, QCEG entropy curves, token CSVs, and overlap/debug reports. It is intentionally isolated from formal evaluation.

## Clean Evaluation Notes

For performance metrics, configure D-squared and question positions in the evaluation wrapper, then keep debug disabled:

```python
model.config.use_d_squared = True
model.config.d_squared_inplace = True
model.config.d_squared_second_stage_method = "qceg"
model.config.d_squared_qceg_tau = 0.1
model.config.d_squared_query_score_mode = "question_only"
model.config.d_squared_dump_selection_debug = False
model.config.d_squared_static_kv_cache = True
model.model.d_squared_question_query_positions = question_positions
model.model.reset_d_squared()
```

The wrapper must set `d_squared_question_query_positions` for each sample after prompt construction and image-token span mapping. If fewer than two question positions are available, the selector falls back to the attention-based path instead of crashing.

