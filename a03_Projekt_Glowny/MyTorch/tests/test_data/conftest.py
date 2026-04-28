import logging
from typing import TypeAlias

import numpy as np
import pytest

from data.batch import BatchProcessing
from data.cache import CacheManager

MtxList: TypeAlias = list[np.ndarray]


################# test batch


@pytest.fixture(scope="function")
def mock_mtx_list() -> MtxList:
    data = [np.zeros((4, 4), dtype=np.uint8) for _ in range(5)]
    logging.getLogger("logger").debug(
        f"[fixture:mock_mtx_list] Generated {len(data)} matrices 4x4"
    )
    return data


@pytest.fixture(scope="function")
def test_paths() -> list[str]:
    return ["test_image_1.png", "test_image_2.png"]


@pytest.fixture(scope="class")
def batch_proc():
    return BatchProcessing()


import logging

import numpy as np

#################### test cache
import pytest

from data.cache import CacheManager


@pytest.fixture(scope="class")
def cache_manager():
    return CacheManager()


@pytest.fixture(scope="function")
def cache_data_sets():
    logger = logging.getLogger("logger")
    valid_data = [np.random.rand(5, 5) for _ in range(2)]
    mixed_data = [np.random.rand(5, 5), np.array([])]
    logger.debug(
        f"[fixture:cache_data_sets] Prepared data: valid={len(valid_data)}, mixed={len(mixed_data)}"
    )
    return {"valid": valid_data, "mixed": mixed_data}


################# dataset
@pytest.fixture(scope="function")
def mock_dataset_data() -> list[np.ndarray]:
    """Generuje uniwersalną listę macierzy do testów Dataset."""
    data = [np.ones((3, 3)) * i for i in range(5)]
    logging.getLogger("logger").debug(
        f"[fixture:mock_dataset_data] Prepared {len(data)} matrices."
    )
    return data
