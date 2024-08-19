import functools
import logging
from dataclasses import dataclass
import typing as t

import numpy as np
from scipy import stats

from . import config
from .utils import seq


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, eq=True)
class Sig:
    value: t.Union[float, t.Tuple[float, ...]]
    baseline: float
    p_sig: float = config.P_SIG

    @property
    def n(self) -> int:
        return len(self.values)

    def __post_init__(self):
        if not (0 < self.p_sig < 1):
            raise ValueError(
                f"p-sig must be between 0 and 1. the provided value is {self.p_sig}"
            )
        if self.n == 0:
            raise ValueError("Sig cannot take an empty sequence.")
        if self.p_sig > config.P_SIG_WARNING:
            _LOGGER.warning(f"p-sig is higher than {config.P_SIG_WARNING}")

    @property
    def values(self) -> t.List[float]:
        return seq(self.value)

    @property
    @functools.lru_cache()
    def p(self) -> float:
        n = len(self.values)
        if n == 1:
            return np.nan
        return stats.ttest_ind(
            a=self.values,
            b=[self.baseline] * self.n,
            equal_var=False,
            alternative="greater",
        ).pvalue

    @property
    def is_stat_sig(self) -> bool:
        return False if np.isnan(self.p) else self.p < self.p_sig

    @property
    def mean(self) -> float:
        return np.mean(self.values).item()

    @property
    def std(self) -> float:
        return np.std(self.values).item()
