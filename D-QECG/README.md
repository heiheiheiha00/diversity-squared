# D² QCEG-IWFC

This repository now uses the same deployment shape as Nuwa: a small editable
Python package injects acceleration into an already loaded model, while model
weights remain owned by the environment.  The repository ships fixed LLaVA and
Transformers source snapshots for offline deployment; the runtime installs both
snapshots without dependencies, while their Python dependencies come from the
PyPI mirror.

## Supported models

- LLaVA-1.5-7B (exact 32-layer paper schedule)
- Qwen2.5-VL (3B/7B and other decoder depths through a depth-normalized schedule)

The public API mirrors Nuwa's wrapper style:

```python
from dqecg import dqecg

model = dqecg(model, budget=128, architecture="auto")
```

The wrapper is instance-local: it does not replace global Transformers classes.
It retains the existing acceleration details:

1. deterministic farthest-point pivots and full-coverage cluster means;
2. post-image user-prompt normalized-cosine entropy with `eps=1e-6`;
3. positive entropy-drop ranking at the first selection boundary;
4. QCEG-weighted cosine feature coverage at the second boundary;
5. physical hidden-state, mask, position, visual-map and KV-cache synchronization;
6. removal of all remaining visual tokens at the final boundary;
7. stable, sorted token indices and an inference-only guard.

For 32-layer LLaVA, the formal schedules are exactly:

| B | N | K9 | K13 |
|---:|---:|---:|---:|
| 192 | 542 | 223 | 34 |
| 128 | 361 | 154 | 21 |
| 64 | 180 | 74 | 12 |

Qwen2.5-VL retains the same stage ratios and scales the 9/13/24 boundaries to
its decoder depth. Its entry count is selected to make the measured layer-average
budget closest to `B`.

## Server environment

Create the isolated environment after copying the repository, the fixed LLaVA
source, and the CUDA wheelhouse to the server:

```bash
bash server/create_dqecg_env.sh
```

The environment is pinned to PyTorch 2.6.0/CUDA 12.4, Transformers 4.54.0 and
lmms-eval 0.3.4. No GitHub or Hugging Face download is performed during setup:
the fixed LLaVA commit `c121f0432da27facab705978f83c4ada465e46fd` must be
uploaded under `src/LLaVA/`, and matching CPython-3.10 Linux CUDA wheels must
be uploaded under `wheelhouse/`. Ordinary Python dependencies are retrieved
from the configurable PyPI mirror.

Run LLaVA MME:

```bash
MODEL_FAMILY=llava MODE=dqecg D2_BUDGET=128 bash server/run_mme.sh
```

The D-QECG adapter accepts `D2_ATTN_IMPLEMENTATION=auto|sdpa|eager|flash_attention_2`.
`auto` prefers an installed FlashAttention-2 build and otherwise uses native
PyTorch SDPA. See `server/README.md` for the ABI-checked offline wheel workflow.

Run Qwen2.5-VL MME:

```bash
MODEL_FAMILY=qwen2_5_vl \
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
MODE=dqecg D2_BUDGET=128 \
bash server/run_mme.sh
```

The D-QECG adapters are registered as `llava_dqecg` and
`qwen2_5_vl_dqecg` in lmms-eval 0.3.4.
