# Official lmms-eval on AutoDL

This folder keeps the integration thin: clone the official
`EvolvingLMMs-Lab/lmms-eval` repo on AutoDL at the compatible `v0.1.0` tag,
install this repository's local LLaVA and transformers forks, then patch the
official LLaVA wrapper so it can forward D-squared arguments and set QCEG
question-token positions.

The first target is LLaVA-1.5-7B on MME.

## 1. Put files on AutoDL

Example layout:

```bash
/root/autodl-tmp/D_squared_method
/root/autodl-tmp/models/llava-v1.5-7b
/root/autodl-tmp/lmms-eval-0.1.0
```

If your paths differ, set `D2_ROOT`, `MODEL_PATH`, and `LMMS_EVAL_DIR` before
running the scripts.

## 2. Install and patch

```bash
cd /root/autodl-tmp/D_squared_method

export D2_ROOT=/root/autodl-tmp/D_squared_method
export LMMS_EVAL_DIR=/root/autodl-tmp/lmms-eval-0.1.0
export GITHUB_PROXY_PREFIX=https://ghfast.top/
export HF_ENDPOINT=https://hf-mirror.com

bash autodl_lmms/setup_autodl.sh
```

The setup script:

- clones official `EvolvingLMMs-Lab/lmms-eval` at `v0.1.0` if it is missing;
- installs local `src/LLaVA` and local `src/transformers` with `--no-deps`;
- installs official `lmms-eval` using
  `autodl_lmms/constraints_d2_llava_lmms.txt` so torch/transformers/accelerate
  stay on the LLaVA-1.5 stack;
- patches official `lmms_eval/models/llava.py` for `v0.1.0`;
- runs a syntax check on the patched wrapper.

By default, GitHub clone uses:

```bash
git clone --branch v0.1.0 --depth 1 \
  https://ghfast.top/https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

Override it if that mirror is unavailable:

```bash
GITHUB_PROXY_PREFIX=https://your-mirror.example/ bash autodl_lmms/setup_autodl.sh
```

## 3. Run official baseline

```bash
cd /root/autodl-tmp/D_squared_method

export MODEL_PATH=/root/autodl-tmp/models/llava-v1.5-7b
export MODE=baseline
export HF_ENDPOINT=https://hf-mirror.com

bash autodl_lmms/run_llava15_7b_mme.sh
```

For a smoke test:

```bash
LIMIT=20 MODE=baseline bash autodl_lmms/run_llava15_7b_mme.sh
```

## 4. Run D-squared QCEG

```bash
cd /root/autodl-tmp/D_squared_method

export MODEL_PATH=/root/autodl-tmp/models/llava-v1.5-7b
export MODE=d_squared
export HF_ENDPOINT=https://hf-mirror.com

bash autodl_lmms/run_llava15_7b_mme.sh
```

Default D-squared settings:

```text
d_squared_image_token_length=576
d_squared_visual_keep_count=32
d_squared_llm_keep_count=32
d_squared_skip_visual_selector=false
d_squared_agg_layer=3
d_squared_second_stage_method=qceg
d_squared_qceg_tau=0.1
d_squared_query_score_mode=question_only
d_squared_static_kv_cache=true
```

Override them with environment variables, for example:

```bash
D2_VISUAL_KEEP=64 D2_LLM_KEEP=64 MODE=d_squared bash autodl_lmms/run_llava15_7b_mme.sh
```

## 5. Run LLM-selector-only Ablation

This skips the first visual selector and lets the LLM-side selector choose 64
visual tokens directly from the full image-token block:

```bash
MODE=llm_selector MODEL_PATH=/root/autodl-tmp/models/llava-v1.5-7b bash autodl_lmms/run_llava15_7b_mme.sh
```

Expected model args include:

```text
d_squared_skip_visual_selector=true
d_squared_visual_keep_count=0
d_squared_llm_keep_count=64
```

## 6. Outputs

By default, results are written to:

```bash
/root/autodl-tmp/lmms-eval-0.1.0/logs
```

Change it with:

```bash
export OUTPUT_PATH=/root/autodl-tmp/lmms-eval-0.1.0/logs_mme_llava15
```

## Notes

- Do not install current `main` / latest PyPI `lmms-eval` into this environment.
  Versions from `0.2.0` onward require newer torch/transformers/accelerate.
- If the run says `Attempted to load model 'llava', but no model for this name
  found`, rerun `bash autodl_lmms/setup_autodl.sh`. The setup patch forces
  `lmms-eval 0.1.0` to import and register only the LLaVA wrapper, instead of
  silently swallowing optional-model import errors.
- If the run warns that `pytorch_model*.bin` files are 0 bytes, wait for the
  model upload to finish before evaluating.
- Keep `--batch_size 1` for D-squared; this code path supports one image sample
  at a time.
- D-squared QCEG defaults to static KV-cache token drop: `use_cache=true`,
  `d_squared_static_kv_cache=true`, and `output_attentions=false`. It selects
  visual tokens during prefill, then reuses the same selected set for decode.
- Set `D2_USE_CACHE=false D2_STATIC_KV_CACHE=false` to reproduce the legacy
  no-cache path for an A/B comparison.
- MME is loaded by official lmms-eval. The scripts default to
  `HF_ENDPOINT=https://hf-mirror.com`; change it if your AutoDL image uses a
  different Hugging Face mirror/cache.
