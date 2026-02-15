import inspect
import logging
import types
from functools import wraps
from time import perf_counter

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
        level=logging.DEBUG,
        format="[%(name)s] %(message)s",
        handlers=[
            RichHandler(
                rich_tracebacks=True, show_time=True, omit_repeated_times=False
            ),
            logging.FileHandler(file_name),
        ],
    )


def _format_args(sig, *args, **kwargs):
    if not sig:
        return f"args: {args}, kwargs: {kwargs}"
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        filtered = {}
        for k, v in bound.arguments.items():
            if k in ("self", "cls"):
                continue

            if hasattr(v, "shape") and hasattr(v, "ndim") and v.ndim > 0:
                filtered[k] = f"Array{v.shape}"

            elif hasattr(v, "item") and hasattr(v, "ndim") and v.ndim == 0:
                filtered[k] = v.item()

            elif callable(v):
                filtered[k] = f"func:{getattr(v, '__name__', 'lambda')}"

            else:
                filtered[k] = v

        return filtered
    except Exception:
        return "error parsing args"


def autologger(func):
    module_name = func.__module__
    func_name = func.__name__
    line_def = getattr(getattr(func, "__code__", {}), "co_firstlineno", 0)

    sig = None
    try:
        sig = inspect.signature(func)
    except Exception:
        pass

    @wraps(func)
    def wrapper(*args, **kwargs):
        instance = args[0] if args and hasattr(args[0], "logger") else None
        logger = instance.logger if instance else logging.getLogger(module_name)

        formatted_args = _format_args(sig, *args, **kwargs)
        logger.info(f"[{func_name}] [line:{line_def}] Started with: {formatted_args}")

        start = perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = perf_counter() - start
            logger.info(f"[{func_name}] [line:{line_def}] Completed in {duration:.4f}s")
            return result
        except Exception as e:
            logger.error(f"[{func_name}] [line:{line_def}] Failed: {str(e)}")
            raise

    return wrapper


def class_autologger(cls):
    cls.logger = logging.getLogger(cls.__module__)

    for name, attr in list(vars(cls).items()):
        if isinstance(attr, (classmethod, staticmethod)):
            original_func = attr.__func__
            setattr(cls, name, type(attr)(autologger(original_func)))
        elif callable(attr) and not name.startswith("__"):
            setattr(cls, name, autologger(attr))

    return cls
