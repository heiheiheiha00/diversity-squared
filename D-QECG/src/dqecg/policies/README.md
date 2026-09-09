# Architecture policy boundary

The formal LLaVA and Qwen2.5-VL methods intentionally own separate policy
modules.

- `llava.py` is the only formal source for LLaVA pruning layers, entropy
  windows, and schedule derivation.
- `qwen2_5_vl.py` is the only formal source for Qwen2.5-VL-7B pruning layers,
  entropy windows, and schedule derivation.
- `base.py` contains only the small read-only interface consumed by the shared
  physical-pruning decoder. It contains no architecture defaults.

Architecture experiments must patch or wrap their matching policy/adapter
inside the experiment process. They must not change the other architecture's
policy module.
