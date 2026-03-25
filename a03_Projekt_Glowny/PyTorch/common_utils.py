import inspect
import logging
import platform
from functools import wraps
from time import perf_counter
from typing import Any, Callable, ParamSpec, cast

import psutil
from rich.logging import RichHandler

from PyTorch import ClassType, MetricsDict, T

P = ParamSpec("P")


def log_system_info() -> None:
    log = logging.getLogger("NeuralRecognizer")

    total_ram = round(psutil.virtual_memory().total / (1024**3), 2)
    sys_info = (
        "[bold cyan]" + "=" * 40 + "\n"
        "[bold white] SYSTEM ENVIRONMENT INFO [/bold white]\n"
        "[bold cyan]" + "=" * 40 + "\n"
        f"OS: {platform.system()} {platform.release()} ({platform.architecture()[0]})\n"
        f"CPU: {platform.processor()}\n"
        f"RAM: {total_ram} GB\n"
        f"Python: {platform.python_version()}\n"
        "[bold cyan]" + "=" * 40 + "[/bold cyan]"
    )

    log.info(sys_info)


def setup_logging(
    level: int = logging.DEBUG,
    file_name: str = "engine_history.log",
    log_to_file: bool = True,
) -> None:
    FILE_FORMAT = "%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    active_handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, markup=True, show_path=True)
    ]
    if log_to_file:
        file_h = logging.FileHandler(file_name, encoding="utf-8")
        file_h.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
        active_handlers.append(file_h)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=active_handlers,
        force=True,
    )
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))


def _format_args(sig: inspect.Signature, *args: Any, **kwargs: Any) -> MetricsDict:
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        filtered = cast(MetricsDict, {})
        for k, v in bound.arguments.items():
            if k in ("self", "cls"):
                continue
            if hasattr(v, "shape") and hasattr(v, "ndim") and getattr(v, "ndim") > 0:
                filtered[k] = f"Array{getattr(v, 'shape')}"
            elif hasattr(v, "item") and hasattr(v, "ndim") and getattr(v, "ndim") == 0:
                filtered[k] = cast(float | int | str, getattr(v, "item")())
            elif callable(v):
                filtered[k] = f"func:{getattr(v, '__name__', 'lambda')}"
            elif isinstance(v, (float, int, str)):
                filtered[k] = v
            else:
                filtered[k] = str(v)
        return filtered
    except Exception:
        return cast(MetricsDict, {"error": "parsing_failed"})


def autologger(func: Callable[P, T]) -> Callable[P, T]:
    module_name: str = func.__module__
    func_name: str = getattr(func, "__name__", str(func))
    line_def: int = getattr(getattr(func, "__code__", {}), "co_firstlineno", 0)

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        instance = args[0] if args and hasattr(args[0], "logger") else None
        logger = getattr(instance, "logger", logging.getLogger(module_name))

        sig = inspect.signature(func)
        formatted_args = _format_args(sig, *args, **kwargs)

        logger.info(
            f"[bold blue]START[/] | [cyan]{func_name}[/] (line:{line_def}) | Args: {formatted_args}"
        )

        start: float = perf_counter()
        try:
            result = func(*args, **kwargs)
            duration: float = perf_counter() - start

            logger.info(
                f"[bold green]DONE[/]  | [cyan]{func_name}[/] | Time: [yellow]{duration:.4f}s[/]"
            )
            return result
        except Exception as e:
            logger.error(
                f"[bold red]FAIL[/]  | [cyan]{func_name}[/] | Reason: {str(e)}",
                exc_info=True,
            )
            raise

    return wrapper


def silent(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return func(*args, **kwargs)

    setattr(wrapper, "_is_silent", True)
    return wrapper


def class_autologger(cls: ClassType) -> ClassType:
    cls_logger = logging.getLogger(cls.__module__)
    setattr(cls, "logger", cls_logger)

    def _wrap(f: Callable[P, T]) -> Callable[P, T]:
        return autologger(f)

    for name, attr in list(vars(cls).items()):
        if isinstance(attr, (classmethod, staticmethod)):
            original_func = attr.__func__
            if not getattr(original_func, "_is_silent", False):
                decorated = _wrap(cast(Callable[..., Any], original_func))
                setattr(cls, name, cast(Any, type(attr))(decorated))

        elif callable(attr) and not name.startswith("__"):
            if not hasattr(attr, "__name__"):
                setattr(attr, "__name__", name)

            if not getattr(attr, "_is_silent", False):
                setattr(cls, name, _wrap(cast(Callable[..., Any], attr)))

    return cast(ClassType, cls)


__all__ = [
    "class_autologger",
    "log_system_info",
    "setup_logging",
    "silent",
]
