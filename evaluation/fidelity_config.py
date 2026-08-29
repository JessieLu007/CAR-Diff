#evaluatiog/fidelity_config.py
from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class EvalConfig:

    default_mode: Literal['all', 'minority'] = 'all'
    minority_col: Optional[str] = None
    minority_val: Any = None
    metric_modes: dict[str, str] = field(default_factory=dict),
    # pre_filtered=True时，default_mode/metric_modes对_subset()不再有意义
    # 但minority_col仍需保留供_active_cols()使用
    pre_filtered: bool = False # 如果数据已经是少数类,就跳过运行时过滤

    def get_mode(self, metric: str) -> str:
        """Return effective mode for a given metric (per-metric override or default)."""
        return self.metric_modes.get(metric, self.default_mode)
