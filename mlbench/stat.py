from dataclasses import dataclass
import typing as t

import numpy as np

from .utils import seq


# TODO: implement
@dataclass(frozen=True, eq=True)
class Sig:
    value: t.Union[float, t.Tuple[float, ...]]
    baseline: float

    @property
    def values(self) -> t.List[float]:
        return seq(self.value)

    @property
    def is_stat_sig(self) -> bool:
        raise NotImplementedError

    @property
    def mean(self) -> float:
        return np.mean(self.values).item()

    @property
    def std(self) -> float:
        return np.std(self.values).item()
