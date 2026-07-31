# D-QECG MME evaluation

## Required files

Verify the local or server copy before running:

```bash
python server/verify_evaluation_bundle.py
```

## Install lmms-eval

The installer downloads the complete `v0.7.1` repository archive through GitHub
mirrors, installs it editable in the isolated Conda environment, and patches its
LLaVA wrapper. It also restores the missing `model_utils` package marker present
in some `v0.7.1` source archives, then verifies the same evaluator import path
used by the real command. Evaluation scripts put the complete checkout first on
`PYTHONPATH`, avoiding incomplete setuptools editable-package maps:

```bash
bash server/install_lmms_eval_only.sh
```

## Compare baseline and D-QECG

Two-sample smoke test:

```bash
GPU_IDS=0 NUM_PROCESSES=1 LIMIT=2 \
  bash server/run_full_mme_comparison.sh
```

Full MME on eight GPUs, matching the standard Accelerate launch style:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
NUM_PROCESSES=8 \
MAIN_PROCESS_PORT=29507 \
  bash server/run_full_mme_comparison.sh
```

Both runs default to the dependency-free `eager` attention backend. To use an
already installed FlashAttention runtime instead:

```bash
ATTN_BACKEND=flash_attention_2 bash server/run_full_mme_comparison.sh
```

Each run writes raw results, logs, and JSON/CSV/Markdown score reports under:

```text
outputs/mme_full_comparison/<timestamp>/
```
