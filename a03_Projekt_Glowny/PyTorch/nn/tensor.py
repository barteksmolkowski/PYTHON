from typing import Annotated, List, Optional, Tuple, TypeAlias

import numpy as np

Mtx: TypeAlias = np.ndarray
T: TypeAlias = np.ndarray
Shape: TypeAlias = Tuple[int, ...]

T2D: TypeAlias = Annotated[np.ndarray, "Batch, Features"]
T3D: TypeAlias = Annotated[np.ndarray, "Batch, Seq_len, Hidden"]
T4D: TypeAlias = Annotated[np.ndarray, "Batch, Channels, Height, Width"]

OList: TypeAlias = Optional[List]
OMtx: TypeAlias = Optional[Mtx]
OT: TypeAlias = Optional[T]
OShape: TypeAlias = Optional[Shape]

OT2D: TypeAlias = Optional[T2D]
OT3D: TypeAlias = Optional[T3D]
OT4D: TypeAlias = Optional[T4D]


def validate_shape(data: np.ndarray, *args) -> None:
    expected = []
    for arg in args:
        if isinstance(arg, tuple) and len(arg) == 2:
            val, count = arg
            expected.extend([val] * count)
        else:
            expected.append(arg)

    if data.ndim != len(expected):
        raise ValueError(
            f"Rank error! {data.ndim}D != {len(expected)}D. Got: {data.shape}"
        )

    for i, target in enumerate(expected):
        if target != -1 and data.shape[i] != target:
            raise ValueError(
                f"Shape mismatch at axis {i}! Received {data.shape[i]}, expected {target}."
            )
