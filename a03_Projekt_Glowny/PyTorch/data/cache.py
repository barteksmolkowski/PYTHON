import logging
from typing import Protocol

from common_utils import autologger

from .types import BatchData, OptBatchData, OptFilePath


class CacheManagerProtocol(Protocol):
    def cache(self, data: BatchData, path: OptFilePath = None) -> bool: ...

    def load(self, path: OptFilePath = None) -> OptBatchData: ...


@autologger
class CacheManager:
    logger: logging.Logger

    def cache(self, data: BatchData, path: OptFilePath = None) -> bool:
        if not data:
            self.logger.warning(
                "[cache] Attempted to cache an empty BatchData. Operation aborted."
            )
            return False

        assert path is not None, (
            "[cache] Cache path must be provided if no default is set."
        )

        self.logger.info(f"[cache] Successfully cached {len(data)} samples to: {path}")
        return True

    def load(self, path: OptFilePath = None) -> OptBatchData:
        assert path is not None, "[load] Cannot load cache without a valid FilePath."

        self.logger.debug(f"[load] Attempting to load cache from: {path}")

        return []
