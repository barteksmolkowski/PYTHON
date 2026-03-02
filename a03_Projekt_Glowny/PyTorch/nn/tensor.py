from typing import Annotated, List, Optional, Tuple, TypeAlias

import numpy as np

Mtx: TypeAlias = np.ndarray
Tensor: TypeAlias = np.ndarray
Shape: TypeAlias = Tuple[int, ...]

Tensor2D: TypeAlias = Annotated[np.ndarray, "Batch, Features"]
Tensor3D: TypeAlias = Annotated[np.ndarray, "Batch, Seq_len, Hidden"]
Tensor4D: TypeAlias = Annotated[np.ndarray, "Batch, Channels, Height, Width"]

OptList: TypeAlias = Optional[List]
OptMtx: TypeAlias = Optional[Mtx]
OptTensor: TypeAlias = Optional[Tensor]
OptShape: TypeAlias = Optional[Shape]

OptTensor2D: TypeAlias = Optional[Tensor2D]
OptTensor3D: TypeAlias = Optional[Tensor3D]
OptTensor4D: TypeAlias = Optional[Tensor4D]


def validate_shape(data: Mtx, expected: Shape) -> None:
    if data.ndim != len(expected):
        raise ValueError(f"Rank error! {data.ndim}D != {len(expected)}D")

    actual = np.array(data.shape)
    target = np.array(expected)

    mask = target != -1
    if not np.array_equal(actual[mask], target[mask]):
        raise ValueError(f"Shape mismatch! Received {data.shape}, expected {expected}")
