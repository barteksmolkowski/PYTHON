from abc import ABC, abstractmethod
from itertools import product
from typing import List, Literal, overload

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .common import TypeMatrix
from .geometry import ImageGeometry


class __ConvolutionActions__(ABC):
    @abstractmethod
    def convolution_2d(
        self, M: TypeMatrix, filtrs: List[TypeMatrix] = None, dilated: int = 1
    ) -> List[TypeMatrix]:
        pass

    @overload
    @abstractmethod
    def apply_filters(
        self,
        channels_or_path: List[TypeMatrix],
        filtr: List[TypeMatrix],
        padding: Literal[True] = True,
    ) -> List[TypeMatrix]: ...

    @overload
    @abstractmethod
    def apply_filters(
        self,
        channels_or_path: List[TypeMatrix],
        filtr: List[TypeMatrix],
        padding: Literal[False],
    ) -> List[TypeMatrix]: ...

    @abstractmethod
    def apply_filters(
        self,
        channels_or_path: List[TypeMatrix],
        filtr: List[TypeMatrix],
        padding: bool = True,
    ) -> List[TypeMatrix]:
        pass


class ConvolutionActions(__ConvolutionActions__):
    def __init__(self):
        self.geometry = ImageGeometry()

    def convolution_2d(
        self, M: TypeMatrix, filtrs: List[TypeMatrix] = None
    ) -> List[TypeMatrix]:

        if filtrs is None or len(filtrs) == 0:
            return [M]

        results = []
        M_np = np.asanyarray(M)
        for filtr in filtrs:
            filtr_np = np.asanyarray(filtr)
            fh, fw = filtr_np.shape
            windows = sliding_window_view(M_np, (fh, fw))

            results.append(np.einsum("ij,klij->kl", filtr_np, windows))
        return results

    def apply_filters(
        self, M_three_channels: List[TypeMatrix], filtrs: List[TypeMatrix]
    ) -> List[TypeMatrix]:

        final_results = []
        for filtr, M in product(filtrs, M_three_channels):
            M_padded = self.geometry.pad(M, pad_value=-1, padding=1)
            conv_res = self.convolution_2d(M_padded, [filtr])
            final_results.extend(conv_res)

        return final_results
