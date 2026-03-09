import logging
from dataclasses import dataclass, field
from typing import Protocol

from common_utils import class_autologger

from .types import BatchData, FilePath, OptBatchData, OptFilePath


class CacheManagerProtocol(Protocol):
    def cache(self, data: BatchData, path: OptFilePath = None) -> bool: ...

    def load(self, path: OptFilePath = None) -> OptBatchData: ...


def load_cache_logic(path: FilePath) -> BatchData:
    return []


def save_cache_logic(data: BatchData, path: FilePath) -> bool:
    if not data:
        return False
    return True


@class_autologger
@dataclass
class CacheManager:
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("CacheManager orchestrator initialized.")

    def load(self, path: FilePath) -> OptBatchData:
        self.logger.debug(f"[load] Attempting to load cache from: {path}")

        try:
            return load_cache_logic(path)
        except Exception as e:
            self.logger.error(f"[load] Failed to load cache from {path}: {e}")
            return None

    def cache(self, data: BatchData, path: FilePath) -> bool:
        if not data:
            self.logger.warning(
                "[cache] Attempted to cache empty data. Operation aborted."
            )
            return False

        success = save_cache_logic(data, path)

        if success:
            self.logger.info(
                f"[cache] Successfully cached {len(data)} samples to: {path}"
            )
        else:
            self.logger.error(f"[cache] Failed to cache data to: {path}")

        return success
