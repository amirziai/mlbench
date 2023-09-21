import typing as t

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
) -> None:
    assert worst <= baseline
    if ax is None:
        _, ax = plt.subplots(figsize=(width, height))
    adj = 1e-2
    plt.xlim([(worst if clip else 0) - adj, 1 + adj])
    if not clip:
        ax.add_patch(Rectangle((0, 0), worst, height, color="silver", alpha=alpha))
    beg = worst
    diff = baseline - worst
    ax.add_patch(Rectangle((beg, 0), diff, height, color="red", alpha=alpha))
    beg += diff
    diff = 1 - baseline + (worst if clip else 0)
    ax.add_patch(Rectangle((beg, 0), diff, height, color="green", alpha=alpha))
    if val is not None:
        vals = val if isinstance(val, t.Collection) else [val]
        plt.scatter(val, [0.5] * len(vals), marker="x", color="k", s=100)
    plt.yticks([])
