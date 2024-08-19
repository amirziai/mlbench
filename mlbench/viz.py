import typing as t

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw(
    worst: float,
    baseline: float,
    val: t.Optional[t.Union[float, t.Collection[float]]] = None,
    clip: bool = False,
    alpha: float = 0.3,
    width: int = 10,
    height: int = 1,
    ax=None,
    p_low: t.Optional[int] = None,
    p_high: t.Optional[int] = None,
) -> None:
    """
    Draw the representation of a metric.
    :param worst: worst possible value for the metric.
    :param baseline: metric baseline.
    :param val: optionally provide a scalar or a collection of scalars to draw.
    :param clip: whether the visualization should start at the `worst` value (clip=True) or at 1.
    :param alpha: alpha for drawing boxes.
    :param width: width of the visualization.
    :param height: height of the visualization.
    :param ax: optional matplotlib ax to pass
    :param p_low: optional percentile (int between 0 and 100) to use for showing the distribution of val.
    ignored if val is None. this is the lower percentile to use. e.g. 25 for the 25th percentile.
    :param p_high: same as above, but the higher percentile. e.g. 75 for the 75th percentile.
    :return: nada
    """
    if not worst <= baseline:
        raise ValueError(f"worst={worst} must be <= baseline={baseline}")
    if ax is None:
        _, ax = plt.subplots(figsize=(width, height))
    adj = 1e-2
    plt.xlim([(worst if clip else 0) - adj, 1 + adj])
    if not clip:
        ax.add_patch(
            Rectangle(
                xy=(0, 0), width=worst, height=height, color="silver", alpha=alpha
            )
        )
    beg = worst
    diff = baseline - worst
    ax.add_patch(
        Rectangle(xy=(beg, 0), width=diff, height=height, color="red", alpha=alpha)
    )
    beg += diff
    diff = 1 - baseline + (worst if clip else 0)
    ax.add_patch(
        Rectangle(xy=(beg, 0), width=diff, height=height, color="green", alpha=alpha)
    )
    if val is not None:
        vals = val if isinstance(val, t.Collection) else [val]
        if p_low is not None:
            if not (0 <= p_low <= 100 and 0 <= p_high <= 100 and p_low <= p_high):
                raise ValueError(
                    "p_low and p_high should be integers between 0 and 100 and p_low <= p_high."
                )
            low = np.percentile(vals, q=int(p_low))
            high = np.percentile(vals, q=int(p_high))
            ax.add_patch(
                Rectangle(
                    xy=(low, 0),
                    width=high - low,
                    height=height,
                    color="blue",
                    alpha=alpha,
                )
            )
        plt.scatter(val, [0.5] * len(vals), marker="x", color="k", s=100)
    plt.yticks([])
