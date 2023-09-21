from dataclasses import dataclass
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import typing as t

import numpy as np

from . import utils, viz, ap
from .stat import Sig


@dataclass(frozen=True, eq=True)
class Metric:
    value: t.Union[float, t.Tuple[float, ...]]
    baseline: float
    min_dataset: float
    min_possible: float
    max_possible: float

    def __post_init__(self):
        # TODO: support lower-is-better metrics
        assert (
            self.min_possible <= self.min_dataset <= self.baseline <= self.max_possible
        )
        for val in self.values:
            assert self.__in_range(val=val)

    def __in_range(self, val: float) -> bool:
        return self.min_dataset <= val <= self.max_possible

    @property
    def stat_sig(self) -> Sig:
        return Sig(value=tuple(self.values), baseline=self.baseline)

    @property
    def values(self) -> t.List[float]:
        return utils.seq(self.value)

    def draw(self) -> None:
        return viz.draw(
            worst=self.min_dataset,
            baseline=self.baseline,
            val=self.value,
        )

    @classmethod
    def from_dataset(
        cls,
        y_true: t.Sequence[bool],
        y_pred: t.Union[
            t.Sequence[float],
            t.Sequence[t.Sequence[float]],
        ],
    ) -> "Metric":
        n = len(y_true)
        assert n > 0, "Input sequences cannot be empty"
        y_pred = [y_pred] if not isinstance(y_pred[0], t.Sequence) else y_pred
        assert all(
            len(yp) == n for yp in y_pred
        ), "all y_pred must be of the same length"
        value = tuple(
            cls._metric(
                y_true=y_true,
                y_pred=yp,
            )
            for yp in y_pred
        )
        return cls._create_cls(value=value, y_true=y_true)

    @staticmethod
    def _metric(y_true: t.Sequence[bool], y_pred: t.Sequence[float]) -> float:
        raise NotImplementedError

    @classmethod
    def _create_cls(
        cls, value: t.Sequence[float], y_true: t.Sequence[bool]
    ) -> "Metric":
        raise NotImplementedError


@dataclass(frozen=True, eq=True)
class AccuracyBinary(Metric):
    min_possible: float = 0
    max_possible: float = 1
    min_dataset: float = 0

    @staticmethod
    def _metric(y_true: t.Sequence[bool], y_pred: t.Sequence[bool]) -> float:
        return accuracy_score(
            y_true=y_true,
            y_pred=y_pred,
        )

    @classmethod
    def _create_cls(
        cls, value: t.Tuple[float, ...], y_true: t.Sequence[bool]
    ) -> "Metric":
        return cls(value=value, baseline=sum(y_true) / len(y_true))


@dataclass(frozen=True, eq=True)
class AUROCBinary(Metric):
    min_possible: float = 0
    max_possible: float = 1
    min_dataset: float = 0
    baseline: float = 0

    @staticmethod
    def _metric(y_true: t.Sequence[bool], y_pred: t.Sequence[bool]) -> float:
        return roc_auc_score(
            y_true=y_true,
            y_score=y_pred,
        )

    @classmethod
    def _create_cls(
        cls, value: t.Tuple[float, ...], y_true: t.Sequence[bool]
    ) -> "Metric":
        return cls(value=value)


@dataclass(frozen=True, eq=True)
class AveragePrecisionBinary(Metric):
    min_possible: float = 0
    max_possible: float = 1

    @staticmethod
    def _metric(y_true: t.Sequence[bool], y_pred: t.Sequence[bool]) -> float:
        return average_precision_score(
            y_true=y_true,
            y_score=y_pred,
        )

    @classmethod
    def _create_cls(
        cls, value: t.Tuple[float, ...], y_true: t.Sequence[bool]
    ) -> "Metric":
        baseline = cls._get_baseline(y_true=y_true)
        min_dataset = cls._get_min(y_true=y_true)
        return cls(value=value, baseline=baseline, min_dataset=min_dataset)

    @staticmethod
    def _get_baseline(y_true: t.Sequence[bool]) -> float:
        n = len(y_true)
        p_ = sum(y_true)
        if n >= 1_000:
            return sum(y_true) / n
        return ap.expected_average_precision(n_=n, p_=p_)

    @staticmethod
    def _get_min(y_true: t.Sequence[bool]) -> float:
        n = len(y_true)
        p_cnt = sum(y_true)
        if n >= 10_000:
            p = p_cnt / n
            q = (1 - p) / p
            s = np.log(1 / (1 - p))
            return 1 - q * s

        return sum(i / (n - p_cnt + i) for i in range(1, p_cnt + 1)) / p_cnt
