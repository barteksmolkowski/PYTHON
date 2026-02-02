import logging
import types

from rich.logging import RichHandler


def build_all(local_vars: dict) -> list[str]:
    base_attrs = dir(types.ModuleType("base"))
    base_attrs.extend(
        [
            "__annotations__",
            "__builtins__",
            "__file__",
            "__cached__",
            "types",
            "__path__",
            "__loader__",
            "__spec__",
            "base_attrs",
        ]
    )

    return [
        n
        for n, obj in local_vars.items()
        if n not in base_attrs
        and not isinstance(obj, types.ModuleType)
        and (n.startswith("__") or "_" in n or n.isupper())
    ]


def setup_logging(file_name="engine_history.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        handlers=[
            RichHandler(rich_tracebacks=True),
            logging.FileHandler(file_name),
        ],
    )
