# D-squared Local Environment

Current scope: only the original LLaVA code path is supported.

Use this environment for:

- `src/LLaVA`
- `src/transformers`
- `src/D_squared_method`

Recommended platform: Linux or WSL2 + CUDA. Native Windows can run some code, but `bitsandbytes` and 4-bit inference are much smoother on Linux/WSL2.

## Setup

```bash
conda create -n d2-llava python=3.10 -y
conda activate d2-llava

cd D_squared_method

pip install --upgrade pip

# CUDA 11.8 build matching the original LLaVA pins.
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Core LLaVA dependencies.
pip install tokenizers==0.13.3 sentencepiece==0.1.99 shortuuid
pip install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.41.0
pip install "pydantic<2,>=1" markdown2[all] numpy scikit-learn==1.2.2
pip install gradio==3.35.2 gradio_client==0.2.9
pip install requests httpx==0.24.0 uvicorn fastapi
pip install einops==0.6.1 einops-exts==0.0.4 timm==0.6.13
pip install datasets pycocoevalcap

# Install local packages. Install LLaVA first, then the modified transformers.
pip install -e src/LLaVA
pip install -e src/transformers

# Make the D_squared_method package importable.
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

On PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
```

## Quick Check

```bash
python - <<'PY'
import torch
import transformers
from D_squared_method.visual_selector import select_visual_token_indices

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print(select_visual_token_indices(torch.randn(16, 8), keep_num=4))
PY
```

## Notes

- Do not install another local `transformers` fork in this environment unless you intentionally want to replace `src/transformers`.
- For 4-bit loading, prefer Linux/WSL2. If `bitsandbytes` fails on native Windows, use full precision/FP16 or move to WSL2.
- Model weights are not included here. Put your LLaVA checkpoint path into the inference scripts with `--model-path`.
