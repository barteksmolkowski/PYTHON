from typing import Tuple, TypeAlias

import numpy as np

Mtx: TypeAlias = np.ndarray
Tensor: TypeAlias = np.ndarray
Shape: TypeAlias = Tuple[int, ...]

def validate_shape(data: Mtx, expected: Shape) -> bool:
    ...
