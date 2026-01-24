# MaxPool2D(M, kernel_size, stride=None, padding = 0)
# // stride(x,y)

from abc import abstractmethod
from typing import List

TypeMatrix = List[List[int]]


class __Pooling__:
    @abstractmethod
    def max_pool(matrix: TypeMatrix, pool_size: int, stride: int):
        pass


class Pooling(__Pooling__):
    @staticmethod
    def max_pool(
        matrix, kernel_size: tuple[int, int], stride=None, padding=0
    ):  # MaxPool2D(M, kernel_size, stride=None, padding = 0)
        def _max_window(M, x, y, kernel_size):
            return max(
                M[y + fy][x + fx]
                for fy in range(kernel_size[0])
                for fx in range(kernel_size[1])
            )

        return [
            [
                _max_window(matrix, x, y, kernel_size)
                for x in range(0, len(matrix[0]) - kernel_size[0] + 1, stride)
            ]
            for y in range(0, len(matrix) - kernel_size[1] + 1, stride)
        ]


import random

# import numpy as np
# matrix = np.random.rand(3,4)
# matrix = np.random.ranomint(1, 11, size=(3,4))

rows = 6
cols = 6
M = [[random.randint(1, 10) for _ in range(cols)] for _ in range(rows)]
print(M)
pooling = Pooling()
wynik = pooling.max_pool(M, (2, 2), 3)
print(wynik)
