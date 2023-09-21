from dataclasses import dataclass
import typing as t

import numpy as np
from scipy import stats

from .utils import seq


# TODO: implement
@dataclass(frozen=True, eq=True)
class Sig:
    value: t.Union[float, t.Tuple[float, ...]]
    baseline: float
    _p: float = 0.05

    @property
    def n(self) -> int:
        return len(self.values)

    def __post_init__(self):
        if self.n == 0:
            raise ValueError("Sig cannot take an empty sequence.")

    @property
    def values(self) -> t.List[float]:
        return seq(self.value)

    @property
    def p(self) -> float:
        return stats.ttest_ind(self.values, [self.baseline] * self.n).pvalue

    @property
    def is_stat_sig(self) -> bool:
        n = len(self.value)
        if n == 1:
            return False
        return self.p < self._p

    @property
    def mean(self) -> float:
        return np.mean(self.values).item()

    @property
    def std(self) -> float:
        return np.std(self.values).item()
