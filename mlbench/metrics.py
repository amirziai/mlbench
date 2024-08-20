"""
Metrics.

TODO: materializing y_true_concat redundantly. See if we can consolidate.
"""

import functools
from dataclasses import dataclass, field

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
import typing as t

import numpy as np

from . import ap, config, utils, viz
from .stat import Sig


Y = t.Union[t.Sequence[float], t.Sequence[t.Sequence[float]]]
Y_BOOL = t.Union[t.Sequence[bool], t.Sequence[t.Sequence[bool]]]


@dataclass(frozen=True, eq=True)
class Metric:
    value: t.Union[float, t.Tuple[float, ...]]
    sample_cnt: int
    baseline: float
    min_dataset: float
    min_possible: float
    max_possible: float
    p_sig: float = config.P_SIG
    precision: int = config.OBJECT_REPR_PRECISION
    object_repr_metric_cnt: float = config.OBJECT_REPR_METRIC_CNT

    def __post_init__(self):
        # TODO: support lower-is-better metrics
        if self.sample_cnt <= 0 or not isinstance(self.sample_cnt, int):
            raise ValueError("sample count must be a positive integer.")
        if not (
            self.min_possible <= self.min_dataset <= self.baseline <= self.max_possible
        ):
            msg = "metric construction check failed."
            msg += f" make sure that min_possible={self.min_possible} <= min_dataset={self.min_dataset}"
            msg += f" <= baseline={self.baseline} <= max_possible={self.max_possible}"
            raise ValueError(msg)
        for val in self.values:
            if not self.__in_range(val=val):
                return f"value={val} is not within the acceptable range [{self.min_dataset}, {self.max_possible}]"

    @property
    def experiment_cnt(self) -> int:
        return len(self.values)

    @staticmethod
    def _pass_emoji(passed: bool) -> str:
        return config.EMOJIS_SUCCESS[passed]

    @property
    def report(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "check": "beat baseline",
                    "pass": self._pass_emoji(passed=self.beat_baseline),
                    "value": self.value_point_estimate,
                    "threshold": self.baseline,
                    "gain": self.value_point_estimate - self.baseline,
                },
                {
                    "check": "stat sig",
                    "pass": self._pass_emoji(passed=self.is_stat_sig),
                    "value": self.stat_sig.p,
                    "threshold": self.p_sig,
                    "gain": self.p_sig - self.stat_sig.p,
                },
                {
                    "check": "experiment count",
                    "pass": self._pass_emoji(
                        passed=self.experiment_cnt >= config.EXPERIMENT_CNT_MIN
                    ),
                    "value": self.experiment_cnt,
                    "threshold": config.EXPERIMENT_CNT_MIN,
                    "gain": self.experiment_cnt - config.EXPERIMENT_CNT_MIN,
                },
                {
                    "check": "data size",
                    "pass": self._pass_emoji(
                        passed=self.sample_cnt >= config.SAMPLE_CNT_MIN
                    ),
                    "value": self.sample_cnt,
                    "threshold": config.SAMPLE_CNT_MIN,
                    "gain": self.sample_cnt - config.SAMPLE_CNT_MIN,
                },
            ]
        ).set_index("check")

    def __in_range(self, val: float) -> bool:
        return self.min_dataset <= val <= self.max_possible

    @property
    @functools.lru_cache()
    def stat_sig(self) -> Sig:
        return Sig(
            value=tuple(self.values),
            baseline=self.baseline,
            p_sig=self.p_sig,
        )

    @property
    @functools.lru_cache()
    def is_stat_sig(self) -> bool:
        return self.stat_sig.is_stat_sig

    @property
    @functools.lru_cache()
    def values(self) -> t.List[float]:
        return utils.seq(self.value)

    def draw(
        self,
        p_low: t.Optional[int] = config.P_LOW,
        p_high: t.Optional[int] = config.P_HIGH,
    ) -> None:
        """
        Draw a representation of the experiment metrics.
        The following optional parameters determine the range of the distribution you want to highlight.
        E.g. use 25 and 75 to draw Inter Quartile Range (IQR) of the experiment metrics.
        :param p_low: int between 0 and 100. e.g. 25 for 25th percentile.
        :param p_high: int between 0 and 100. e.g. 75 for 75th percentile.
        """
        return viz.draw(
            worst=self.min_dataset,
            baseline=self.baseline,
            val=self.value,
            p_high=p_high,
            p_low=p_low,
        )

    @classmethod
    def from_dataset(
        cls,
        y_true: Y,
        y_pred: Y,
        p_sig: float = config.P_SIG,
    ) -> "Metric":
        value = cls._get_value(y_true=y_true, y_pred=y_pred)
        return cls._create_cls(value=value, y_true=y_true, p_sig=p_sig)

    @classmethod
    def _get_value(cls, y_true: Y, y_pred: Y) -> t.Tuple[float, ...]:
        y_true_seq = utils.seq_of_seq(y_true)
        if len(y_true_seq) == 0 or any(len(yt) == 0 for yt in y_true_seq):
            raise ValueError("Input sequences cannot be empty")
        y_pred_seq = utils.seq_of_seq(y_pred)
        if len(y_true_seq) == 1:
            n = len(y_true_seq[0])
            if not all(n == len(yp) for yp in y_pred_seq):
                raise ValueError("all y_pred must be of equal length to y_true")
        else:
            if len(y_pred_seq) != len(y_true_seq):
                raise ValueError(
                    f"y_true ({len(y_true_seq)}) and y_pred ({len(y_pred_seq)}) "
                    f"must have the same number of experiments."
                )
            if not all(len(yt) == len(yp) for yt, yp in zip(y_true_seq, y_pred_seq)):
                pass

        return tuple(
            cls._metric(
                y_true=y_true_seq[0] if len(y_true_seq) == 1 else y_true_seq[idx],
                y_pred=yp,
            )
            for idx, yp in enumerate(y_pred_seq)
        )

    @staticmethod
    def _metric(y_true: t.Sequence[float], y_pred: t.Sequence[float]) -> float:
        raise NotImplementedError

    @classmethod
    def _create_cls(
        cls, value: t.Sequence[float], y_true: t.Sequence[float], p_sig: float
    ) -> "Metric":
        raise NotImplementedError

    @property
    @functools.lru_cache()
    def value_point_estimate(self) -> float:
        fn = np.mean  # TODO: make this a param
        return fn(self.values).item()

    @property
    @functools.lru_cache()
    def beat_baseline(self) -> bool:
        return self.value_point_estimate > self.baseline

    @property
    @functools.lru_cache()
    def beat_pct(self) -> float:
        return 100 * (np.array(self.values) > self.baseline).sum() / len(self.values)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        details = config.EMOJIS_SUCCESS[self.beat_baseline]
        details += f" {self.value_point_estimate:.{self.precision}f} does"
        details += "" if self.beat_baseline else " not"
        details += f" beat the baseline {self.baseline:.{self.precision}f} ({int(self.beat_pct)}%), "
        details += config.EMOJIS_SUCCESS[self.is_stat_sig]
        details += "" if self.is_stat_sig else " not"
        details += f" stat sig"
        metrics_repr = [
            round(v, self.precision) for v in self.values[: self.object_repr_metric_cnt]
        ]
        details += f" , {len(self.values)} experiment metric"
        details += (
            f"s: {metrics_repr}"
            + ("..." if len(self.values) > self.object_repr_metric_cnt else "")
            if len(self.values) > 1
            else f": {metrics_repr[0]}"
        )
        thumb = (
            config.EMOJIS_THUMBS["up"] * min(3, len(str(self.sample_cnt)) - 2)
            if self.sample_cnt >= 100
            else config.EMOJIS_THUMBS["down"]
        )
        details += f", {self.sample_cnt:,} data instances {thumb}"
        return f"{name}({details})"

    @staticmethod
    def _get_sample_cnt(y_true: Y) -> int:
        return min(len(yt) for yt in utils.seq_of_seq(y_true))

    @staticmethod
    def _get_y_true_concat(y_true: Y) -> np.ndarray:
        return np.concatenate(utils.seq_of_seq(y_true))


@dataclass(frozen=True, eq=True, repr=False)
class BinaryMetric(Metric):
    @classmethod
    def from_dataset(
        cls,
        y_true: Y_BOOL,
        y_pred: Y,
        p_sig: float = config.P_SIG,
    ) -> "Metric":
        if not all(
            np.all(np.isin(np.array(yt), [0, 1, True, False]))
            for yt in utils.seq_of_seq(y_true)
        ):
            raise ValueError("All target values passed for y_true must be binary.")
        return super().from_dataset(y_true=y_true, y_pred=y_pred, p_sig=p_sig)


@dataclass(frozen=True, eq=True, repr=False)
class AccuracyBinary(BinaryMetric):
    min_possible: float = 0
    max_possible: float = 1
    min_dataset: float = 0

    @staticmethod
    def _metric(y_true: Y_BOOL, y_pred: Y) -> float:
        return accuracy_score(
            y_true=y_true,
            y_pred=y_pred,
        )

    @classmethod
    def _create_cls(
        cls, value: t.Tuple[float, ...], y_true: t.Sequence[bool], p_sig: float
    ) -> "Metric":
        y_true_concat = cls._get_y_true_concat(y_true=y_true)
        return cls(
            value=value,
            baseline=sum(y_true_concat) / len(y_true_concat),
            p_sig=p_sig,
            sample_cnt=cls._get_sample_cnt(y_true=y_true),
        )


@dataclass(frozen=True, eq=True, repr=False)
class BalancedAccuracyBinary(AccuracyBinary):
    min_possible: float = 0
    max_possible: float = 1
    min_dataset: float = 0
    baseline: float = field(init=False, default=0.5)

    @staticmethod
    def _metric(y_true: Y_BOOL, y_pred: Y) -> float:
        return balanced_accuracy_score(
            y_true=y_true,
            y_pred=y_pred,
        )

    @classmethod
    def _create_cls(
        cls,
        value: t.Tuple[float, ...],
        y_true: t.Sequence[bool],
        p_sig: float,
    ) -> "BalancedAccuracyBinary":
        return cls(
            value=value, p_sig=p_sig, sample_cnt=cls._get_sample_cnt(y_true=y_true)
        )


@dataclass(frozen=True, eq=True, repr=False)
class AUROCBinary(BinaryMetric):
    min_possible: float = 0
    max_possible: float = 1
    min_dataset: float = 0
    baseline: float = field(init=False, default=0.5)

    @staticmethod
    def _metric(y_true: Y_BOOL, y_pred: Y) -> float:
        return roc_auc_score(
            y_true=y_true,
            y_score=y_pred,
        )

    @classmethod
    def _create_cls(
        cls,
        value: t.Tuple[float, ...],
        y_true: t.Sequence[bool],
        p_sig: float,
    ) -> "Metric":
        return cls(
            value=value, p_sig=p_sig, sample_cnt=cls._get_sample_cnt(y_true=y_true)
        )


@dataclass(frozen=True, eq=True, repr=False)
class AveragePrecisionBinary(BinaryMetric):
    min_possible: float = 0
    max_possible: float = 1

    @staticmethod
    def _metric(y_true: Y_BOOL, y_pred: Y) -> float:
        return average_precision_score(
            y_true=y_true,
            y_score=y_pred,
        )

    @classmethod
    def _create_cls(
        cls, value: t.Tuple[float, ...], y_true: Y_BOOL, p_sig: float
    ) -> "Metric":
        # baseline and min values might be different for each data partition
        # this is especially true for smaller datasets
        # we're concatenating all partitions (if more than one) into a larger
        # ground truth label pool to derive a better estimate
        # TODO: the proper way to do this is to make Metric aware of each partition
        # this is largely a problem for metrics such as AP and accuracy that don't have
        # global baseline values
        y_true_concat = cls._get_y_true_concat(y_true=y_true)
        baseline = cls._get_baseline(y_true=y_true_concat)
        min_dataset = cls._get_min(y_true=y_true_concat)
        return cls(
            value=value,
            baseline=baseline,
            min_dataset=min_dataset,
            p_sig=p_sig,
            sample_cnt=cls._get_sample_cnt(y_true=y_true),
        )

    @staticmethod
    def _get_baseline(y_true: t.Sequence[bool]) -> float:
        return ap.expected_average_precision(n_=len(y_true), p_=sum(y_true))

    @staticmethod
    def _get_min(y_true: t.Sequence[bool]) -> float:
        return ap.minimum_average_precision(n_=len(y_true), p_=sum(y_true))
