import typing as t

from . import config
from . import metrics

_METRICS = dict(
    balanced_accuracy=metrics.BalancedAccuracyBinary,
    accuracy=metrics.AccuracyBinary,
    average_precision=metrics.AveragePrecisionBinary,
    auroc=metrics.AUROCBinary,
)


def get_eval_binary_metrics() -> t.Set[str]:
    return set(_METRICS.keys())


def eval_binary(
    y_true: t.Sequence[bool],
    y_pred: metrics.Y_PRED,
    metric: str = "balanced_accuracy",
    p_sig: float = config.P_SIG,
) -> metrics.Metric:
    """
    Run an evaluation for a task with binary ground truth labels.
    :param y_true: a sequence (i.e. list, tuple, or numpy array) of boolean values representing the ground truth labels.
    :param y_pred: either a sequence (parallel to above) with predicted values (bool or float depending on the metric)
    or a sequence of these sequences.
    if it's the latter, each inner sequence needs to be parallel to y_true. Each of these is probably the output of the
    same model across different experiments. E.g. different bootstraps or cross-validation.
    :param metric: name of the metric to use. use the method get_eval_binary_metrics() to get a list of available
    options.
    :param p_sig: threshold for significance testing.
    :return: Metric object that contains results and can be used for visualization and reporting.
    """
    if metric not in _METRICS:
        raise ValueError(
            f"metric={metric} is not supported. pick from {set(_METRICS.keys())}"
        )

    return _METRICS[metric].from_dataset(
        y_true=y_true,
        y_pred=y_pred,
        p_sig=p_sig,
    )
