import functools

from scipy.special import comb as scipy_comb


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


def expected_average_precision(n_: int, p_: int) -> float:
    assert 0 < p_ <= n_
    if p_ == n_:
        return 1

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
