from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional


class BasePreset(ABC):
    def __init__(
        self,
        random_state: Optional[int] = None,
        auto_report: bool = True,
        minimal_report: bool = False,
    ):
        self.random_state = random_state
        self.auto_report = auto_report
        self.minimal_report = minimal_report

    @abstractmethod
    def generate(
        self, data: pd.DataFrame, n_samples: int, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        pass
