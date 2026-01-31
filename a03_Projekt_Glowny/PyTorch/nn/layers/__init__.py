import types

from .a1 import *
from .b2 import *
from .c3 import *

# zmienic nazwy na prawidłowe warstwy ai

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
    ]
)

imported_names = [
    n
    for n in locals()
    if n not in base_attrs and n != "base_attrs" and (n.startswith("__") or "_" in n)
]

__all__ = imported_names

if "base_attrs" in locals():
    del base_attrs
if "imported_names" in locals():
    del imported_names