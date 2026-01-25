from typing import Any, Literal, Protocol, Tuple, overload

import numpy as np

Mtx = np.ndarray
Range = Tuple[float, float]

class NormalizationProtocol(Protocol):
    def normalize(self, M: Mtx, old_r: Range = (0, 255), new_r: Range = (0, 1)) -> Mtx: ...
    
    def z_score_normalization(self, M: Mtx) -> Mtx: ...

    @overload
    def process(self, M: Mtx, use_z_score: Literal[True]) -> Mtx: ...
    
    @overload
    def process(self, M: Mtx, use_z_score: Literal[False], old_r: Range, new_r: Range) -> Mtx: ...
    
    def process(self, M: Mtx, use_z_score: bool, old_r: Any, new_r: Any) -> Mtx: ...


class Normalization:
    def normalize(self, M: Mtx, old_r: Range = (0, 255), new_r: Range = (0, 1)) -> Mtx:
        denominator = old_r[1] - old_r[0]
        
        if denominator == 0:
            return np.zeros_like(M, dtype=np.float32) + new_r[0]
            
        return (M - old_r[0]) * (new_r[1] - new_r[0]) / denominator + new_r[0]

    def z_score_normalization(self, M: Mtx) -> Mtx:
        mu = np.mean(M)
        sigma = np.std(M)
        
        eps = 1e-8
        return (M - mu) / (sigma + eps)

    @overload
    def process(self, M: Mtx, use_z_score: Literal[True]) -> Mtx: ...

    @overload
    def process(self, M: Mtx, use_z_score: Literal[False], old_r: Range, new_r: Range) -> Mtx: ...

    def process(
        self, 
        M: Mtx, 
        use_z_score: bool = True, 
        old_r: Range = (0, 255), 
        new_r: Range = (0, 1)
    ) -> Mtx:
        M_np = np.asanyarray(M, dtype=np.float32)
        
        if use_z_score:
            return self.z_score_normalization(M_np)
        
        return self.normalize(M_np, old_r, new_r)
    