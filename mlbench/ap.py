import functools
import logging

import numpy as np
from scipy.special import comb as scipy_comb


_LOGGER = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1_000_000)
def _comb(a: int, b: int):
    return scipy_comb(a, b)


def _n_choose_k(a: int, b: int):
    if b > a:
        return 0
    if a == b == 0:
        return 1
    if a == 0:
        return 0
    if a == b:
        return 1
    return _comb(a, min(b, a - b))


def _checks_passed(n_: int, p_: int) -> bool:
    if p_ == 0:
        _LOGGER.warning(
            "average precision is not defined for a dataset with no positive instances."
        )
        return False
    if p_ > n_:
        raise ValueError(
            f"doesn't make sense to have more positives ({p_}) than there are data points ({n_})"
        )
    return True


def expected_average_precision(n_: int, p_: int) -> float:
    """
    Computes expected average precision (AP) that will be used as a baseline.
    This is the average precision you'd expect if you were to use random rankings.
    Details: https://medium.com/@amirziai/ranking-metrics-from-first-principles-average-precision-32ad65fd18b6
    :param n_: number of data points in the dataset
    :param p_: number of positive data points in the dataset. 0 < p_ <= n.
    :return: expected AP.
    """
    passed = _checks_passed(n_=n_, p_=p_)
    if not passed:
        return np.nan
    if p_ == n_:
        return 1
    if n_ >= 1_000:
        # tends to p_ / n_ as n_ grows
        return p_ / n_

    tot = 0
    cnt_running = 0

    for i in range(1, p_ + 1):
        for n in range(i, n_ - p_ + i + 1):
            bef = (
                1
                if i == 1
                else _n_choose_k(
                    n - 1,
                    i - 1,
                )
            )
            if bef == 0:
                continue
            aft = (
                1
                if i == p_
                else _n_choose_k(
                    n_ - n,
                    p_ - i,
                )
            )
            if aft == 0:
                continue
            cnt = bef * aft
            cnt_running += cnt
            contrib = i / n
            cc = cnt * contrib
            tot += cc

    return tot / cnt_running


def minimum_average_precision(n_: int, p_: int) -> float:
    """
    Computes minimum possible average precision (AP) given a dataset.
    Note that this is not necessarily 0.
    Details: https://medium.com/@amirziai/ranking-metrics-from-first-principles-average-precision-32ad65fd18b6
    :param n_: number of data points in the dataset
    :param p_: number of positive data points in the dataset. 0 < p_ <= n.
    :return: minimum / worst-case AP given a dataset with n_ samples, p_ of which are positive.
    """
    passed = _checks_passed(n_=n_, p_=p_)
    if not passed:
        return np.nan
    if n_ >= 10_000:
        p = p_ / n_
        q = (1 - p) / p
        s = np.log(1 / (1 - p))
        return 1 - q * s
    else:
        return sum(i / (n_ - p_ + i) for i in range(1, p_ + 1)) / p_
