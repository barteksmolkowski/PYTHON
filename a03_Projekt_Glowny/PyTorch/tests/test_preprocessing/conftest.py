import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ENGINES = {
    "ga_engine": ("GeometryAugmentation", "class"),
    "noise_engine": ("NoiseAugmentation", "class"),
    "morph_engine": ("MorphologyAugmentation", "class"),
    "data_aug_engine": ("DataAugmentation", "class"),
    "converter_engine": ("ImageToMatrixConverter", "class"),
    "geometry_engine": ("ImageGeometry", "class"),
    "grayscale_engine": ("GrayScaleProcessing", "class"),
    "image_handler": ("ImageHandler", "class"),
    "pooling_engine": ("Pooling", "class"),
    "thresholding_engine": ("Thresholding", "class"),
    "norm_engine": ("Normalization", "class"),
    "idp_engine": ("ImageDataPreprocessing", "class"),
    "pipeline_engine": ("TransformPipeline", "class"),
}

MTX_CONFIGS = {
    "mock_rgb": ((10, 10, 3), "function"),
    "mock_gray": ((10, 10), "function"),
    "mock_rgba": ((10, 10, 4), "function"),
    "mock_img_data": ((10, 20, 3), "function"),
    "mock_geometry_mtx": ((50, 50), "function"),
    "mock_28x28_sample": ((28, 28), "function"),
    "mock_empty_mtx": ({"data": [[]]}, "function"),
    "mock_norm_mtx": ({"data": [[10, 20], [30, 40]]}, "function"),
    "mock_precision_rgb": ({"data": [[[255, 0, 0], [0, 255, 0]]]}, "function"),
    "decorator_mtx": ({"data": [[1, 0], [0, 1]]}, "function"),
    "mock_pooling_mtx": (
        {"data": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]},
        "function",
    ),
}


def register_fixtures(config_dict, is_engine=True):
    for fix_name, (value, scp) in config_dict.items():

        def factory(val=value, eng=is_engine):
            @pytest.fixture(scope=scp)
            def dynamic_fix():
                if eng:
                    import preprocessing

                    obj = getattr(preprocessing, val)()
                    for attr in [
                        "geometry",
                        "converter",
                        "pipeline",
                        "grayscale",
                        "augmentation",
                        "normalization",
                    ]:
                        if hasattr(obj, attr):
                            setattr(obj, attr, MagicMock())
                    return obj
                return (
                    np.array(val["data"], dtype=np.uint8)
                    if isinstance(val, dict)
                    else np.ones(val, dtype=np.uint8)
                )

            return dynamic_fix

        globals()[fix_name] = factory()


register_fixtures(ENGINES, is_engine=True)
register_fixtures(MTX_CONFIGS, is_engine=False)


@pytest.fixture(scope="session", autouse=True)
def configure_test_logger():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )


@pytest.fixture(scope="function")
def mock_mtx():
    M = np.zeros((28, 28), dtype=np.uint8)
    M[10:20, 10:20] = 255
    return M


@pytest.fixture(scope="function")
def mock_threshold_mtx():
    M = np.full((10, 10), 200, dtype=np.uint8)
    M[4:6, 4:6] = 50
    return M


@pytest.fixture(scope="function")
def mock_io():
    rgb = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    with patch("PIL.Image.open") as m_open, patch("numpy.array", return_value=rgb):
        m_img = MagicMock()
        m_img.convert.return_value = m_img
        m_img.__enter__.return_value = m_img
        m_open.return_value = m_img
        yield rgb


@pytest.fixture(scope="class")
def mock_base():
    return type("MockBase", (), {"method": lambda s, M, *a, **k: (M, a, k)})()


@pytest.fixture(scope="function")
def mock_batch():
    return [np.ones((28, 28), dtype=np.uint8)]
