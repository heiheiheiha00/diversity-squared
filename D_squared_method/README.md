# D-squared Method

D-squared is implemented on the original LLaVA path only. The pruning switch is off by default, so the same environment can run vanilla LLaVA for comparison.

## Setup

See `ENV_SETUP.md` for the local environment. In short, install the local original LLaVA package first, then the modified local `transformers` package:

```bash
cd D_squared_method
pip install -e src/LLaVA
pip install -e src/transformers
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

On PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
```

## D-squared Evaluation

We follow FastV's evaluation style: use the official [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) pipeline for benchmark evaluation, and patch the LLaVA model loaded by lmms-eval so it uses the local D-squared LLaVA/transformers implementation.

Detailed steps are in:

```text
src/D_squared_method/lmms-eval/README.md
```

Baseline comparison is supported exactly like FastV:

- Do not set `model.config.use_d_squared`, or set it to `False`, to run vanilla LLaVA.
- Set `model.config.use_d_squared = True` and call `model.model.reset_d_squared()` to enable D-squared.
- Real token dropping requires `d_squared_inplace=True`, `use_cache=False`, and `output_attentions=True`.

The local non-lmms scripts are still available for quick debugging:

```bash
bash ./src/D_squared_method/inference/eval/eval_ocrvqa_d_squared_token_mask.sh
bash ./src/D_squared_method/inference/eval/eval_aokvqa_latency_d_squared_inplace.sh
```

