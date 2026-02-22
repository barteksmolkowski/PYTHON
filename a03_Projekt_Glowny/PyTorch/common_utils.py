import inspect
import logging
import platform
import types
from functools import wraps
from time import perf_counter

import psutil
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


def log_system_info():
    log = logging.getLogger("NeuralRecognizer")

    log.info("[bold cyan]" + "=" * 40)
    log.info("[bold white] SYSTEM ENVIRONMENT INFO [/bold white]")
    log.info("[bold cyan]" + "=" * 40)
    log.info(
        f"OS: {platform.system()} {platform.release()} ({platform.architecture()[0]})"
    )
    log.info(f"CPU: {platform.processor()}")

    total_ram = round(psutil.virtual_memory().total / (1024**3), 2)
    log.info(f"RAM: {total_ram} GB")

    log.info(f"Python: {platform.python_version()}")
    log.info("[bold cyan]" + "=" * 40 + "[/bold cyan]")


def setup_logging(
    level: int = logging.DEBUG, 
    file_name: str = "engine_history.log", 
    log_to_file: bool = True
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
    func_name = getattr(func, "__name__", str(func))
    line_def = getattr(getattr(func, "__code__", {}), "co_firstlineno", 0)

    @wraps(func)
    def wrapper(*args, **kwargs):
        instance = None
        if len(args) > 0:
            first_arg = args[0]
            if hasattr(first_arg, "logger"):
                instance = first_arg

        logger = instance.logger if instance else logging.getLogger(module_name)

        formatted_args = _format_args(inspect.signature(func), *args, **kwargs)

        logger.info(
            f"[bold blue]START[/] | [cyan]{func_name}[/] (line:{line_def}) | Args: {formatted_args}"
        )

        start = perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = perf_counter() - start

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


def silent(func):
    func._is_silent = True
    return func


def class_autologger(cls):
    cls.logger = logging.getLogger(cls.__module__)

    for name, attr in list(vars(cls).items()):
        if isinstance(attr, (classmethod, staticmethod)):
            func = attr.__func__
            if getattr(func, "_is_silent", False):
                continue

            setattr(cls, name, type(attr)(autologger(func)))

        elif callable(attr) and not name.startswith("__"):
            if not hasattr(attr, "__name__"):
                attr.__name__ = name

            if getattr(attr, "_is_silent", False):
                continue

            setattr(cls, name, autologger(attr))

    return cls
