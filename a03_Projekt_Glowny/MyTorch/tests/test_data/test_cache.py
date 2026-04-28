import logging
from typing import TypeAlias

import numpy as np
from common_utils import class_autologger, silent

MtxList: TypeAlias = list[np.ndarray]


@class_autologger
class TestCacheManager:
    logger: logging.Logger

    def test_cache_empty_input(self, cache_manager) -> None:
        data: list = []
        result = cache_manager.cache(data)

        if result is not False:
            self.logger.error(
                f"[test_cache_empty_input] Logic failed: expected False, got {result}."
            )

        self.logger.info(f"[test_cache_empty_input] Verified empty list rejection.")
        assert result is False

    def test_cache_with_empty_matrix(self, cache_manager, cache_data_sets):
        mixed_data = cache_data_sets["mixed"]
        result = cache_manager.cache(mixed_data)

        if result:
            self.logger.debug(
                f"[test_cache_with_empty_matrix] Cache operation returned True for mixed data."
            )

        assert result is True

    def test_load_empty_source(self, cache_manager):
        loaded = cache_manager.load()

        if isinstance(loaded, list) and len(loaded) == 0:
            self.logger.warning(f"[test_load_empty_source] load() returned empty list.")

        assert loaded == []

    @silent
    def test_cache_success_flow(self, cache_manager, cache_data_sets):
        valid_data = cache_data_sets["valid"]
        result = cache_manager.cache(valid_data)

        if not result:
            self.logger.error(f"[test_cache_success_flow] Failed to cache valid data.")

        assert result is True
