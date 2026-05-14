# D-squared with Official LMMs-Eval

This follows the same integration idea as FastV: use the official `lmms-eval` repository, but make its LLaVA model import the local D-squared LLaVA and modified local `transformers`.

## 1. Prepare D-squared LLaVA

```bash
cd D_squared_method
pip install -e src/LLaVA
pip install -e src/transformers
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

PowerShell:

```powershell
cd D_squared_method
pip install -e src/LLaVA
pip install -e src/transformers
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
```

## 2. Install Official LMMs-Eval

Use the official evaluation framework:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install -e .
```

The official command shape is:

```bash
accelerate launch --num_processes=1 -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1_5_7b_baseline \
    --output_path ./logs/
```

## 3. Patch LMMs-Eval's LLaVA Wrapper

Open the official lmms-eval file:

```text
lmms_eval/models/llava.py
```

After lmms-eval loads the LLaVA model with `load_pretrained_model(...)`, add the D-squared config block below. This is the only evaluation-side patch needed because D-squared is already inserted into local LLaVA and local `transformers`.

For vanilla baseline, use:

```python
self._model.config.use_d_squared = False
self._model.model.reset_d_squared()
```

For D-squared token dropping, use:

```python
self._model.config.use_d_squared = True
self._model.config.d_squared_inplace = True
self._model.config.d_squared_sys_length = 36
self._model.config.d_squared_image_token_length = 576
self._model.config.d_squared_visual_keep_count = 144
self._model.config.d_squared_llm_keep_count = 144
self._model.config.d_squared_agg_layer = 3
self._model.model.reset_d_squared()
```

Use `d_squared_agg_layer = 3` to match FastV's single-layer pruning position. Change the two keep-count values for ablations.

## 4. Ensure Generation Uses D-squared Requirements

D-squared token-drop mode needs attention from the previous layer and cannot keep the original KV cache after sequence pruning. In the lmms-eval LLaVA wrapper, make sure the `generate(...)` call uses:

```python
use_cache=False
output_attentions=True
```

For baseline vanilla LLaVA, these are not required, but keeping the same generation settings makes comparisons cleaner.

## 5. Run Official LMMs-Eval

Example baseline:

```bash
accelerate launch --num_processes=1 -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1_5_7b_baseline \
    --output_path ./logs/
```

Example D-squared run after applying the D-squared config block:

```bash
accelerate launch --num_processes=1 -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1_5_7b_d_squared_k3_v144_l144 \
    --output_path ./logs/
```

You can replace `mme` with any task supported by official lmms-eval, for example `gqa`, `mmmu_val`, `seedbench`, or a comma-separated task list.

