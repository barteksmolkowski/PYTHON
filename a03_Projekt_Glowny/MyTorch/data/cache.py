from dataclasses import dataclass

from data.base import CacheManagerABC
from MyTorch import BatchData, FilePath


def load_cache_logic(path: FilePath) -> BatchData:
    return []


def save_cache_logic(data: BatchData, path: FilePath) -> bool:
    if not data:
        return False
    return True


@dataclass(frozen=True, slots=True)
class CacheManager(CacheManagerABC):
    def load(self, path: FilePath) -> BatchData:
        try:
            return load_cache_logic(path)
        except Exception as e:
            self.logger.error(f"Failed to load cache from {path}: {e}")
            return []

    def cache(self, data: BatchData, path: FilePath) -> bool:
        if not data:
            return False
        return save_cache_logic(data, path)
