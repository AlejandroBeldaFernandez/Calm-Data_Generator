from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class DateConfig:
    """
    Configuration for timestamp injection in generated data.
    """

    date_col: str = "timestamp"
    start_date: Optional[str] = None
    frequency: int = 1
    step: Optional[Dict[str, int]] = None  # e.g. {"days": 1}


@dataclass
class DriftConfig:
    """
    Configuration for drift injection.
    This is a high-level wrapper to standardize how drift is passed.
    """

    drift_type: str = "none"
    position: Optional[int] = None
    width: Optional[int] = None
    options: Dict = field(default_factory=dict)
