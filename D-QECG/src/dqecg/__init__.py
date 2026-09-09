"""D-squared QCEG-IWFC project utilities.

The public Python package is named ``dqecg`` because Python module names
cannot contain hyphens. Legacy ``d_squared_*`` model configuration keys remain
unchanged for checkpoint and wrapper compatibility.
"""

from .budget import DQECGBudget, derive_dqecg_budget
from .config import DQECGConfig, DQECGSchedule, derive_schedule
from .iwfc import select_visual_tokens_by_iwfc
from .main import dqecg, dqecg_llava, dqecg_qwen2_5_vl

__all__ = [
    "DQECGBudget",
    "DQECGConfig",
    "DQECGSchedule",
    "derive_dqecg_budget",
    "derive_schedule",
    "dqecg",
    "dqecg_llava",
    "dqecg_qwen2_5_vl",
    "select_visual_tokens_by_iwfc",
]
