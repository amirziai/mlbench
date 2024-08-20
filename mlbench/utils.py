import typing as t

import numpy as np

A = t.TypeVar("A")


def _check_seq(val: t.Any) -> bool:
    return isinstance(val, (t.Sequence, np.ndarray)) and not isinstance(val, str)


def seq(val: t.Union[A, t.Sequence[A]]) -> t.List[A]:
    return val if _check_seq(val) else [val]


def seq_of_seq(
    val: t.Union[t.Sequence[A], t.Sequence[t.Sequence[A]]]
) -> t.List[t.Sequence[A]]:
    if _check_seq(val):
        if len(val) == 0:
            raise ValueError("the input sequence cannot be empty")
        # if a single sequence s, return [s]
        # if a sequence of sequences, return as is
        return val if _check_seq(val[0]) else [val]
    else:
        raise ValueError("You must pass a sequence (list, tuple, or np.array)")
