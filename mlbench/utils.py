import typing as t

A = t.TypeVar("A")


def seq(val: t.Union[A, t.Sequence[A]]) -> t.List[A]:
    return (
        list(val) if isinstance(val, t.Sequence) and not isinstance(val, str) else [val]
    )
