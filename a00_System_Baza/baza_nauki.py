import os
import shutil
from functools import wraps
from typing import Optional

__all__ = ["__BazaNauki__", "bezpieczny_wrapper", "dekoruj_wszystko"]


class __BazaNauki__:
    def __init__(
        self,
        aktywne=True,
        metody: Optional[list] = None,
        na_odwrot_metody: bool = False,
    ):
        if not aktywne:
            print(f">>> SEKCJA POMINIĘTA: {self.__class__.__name__}")
            return

        wszystkie_publiczne = [
            n
            for n in dir(self)
            if callable(getattr(self, n)) and not n.startswith("_") and n != "run"
        ]

        wybrane_metody = metody if metody is not None else wszystkie_publiczne

        print(f"\n>>> URUCHAMIAM SEKCJĘ: {self.__class__.__name__} <<<")

        for nazwa in wszystkie_publiczne:
            wartosc = getattr(self, nazwa)

            czy_uruchomic = (nazwa in wybrane_metody) != na_odwrot_metody

            if czy_uruchomic:
                wartosc()


def bezpieczny_wrapper(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        tmp_folder = "tmp"
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
            print(f"[WRAPPER] Stworzono folder tymczasowy: {tmp_folder}")
        else:
            shutil.rmtree(tmp_folder)
            print(f"[WRAPPER] Usunięto folder tymczasowy: {tmp_folder}")
            os.makedirs(tmp_folder)
            print(f"[WRAPPER] Stworzono folder tymczasowy: {tmp_folder}")

        return func(self, *args, **kwargs)

    return wrapper


def dekoruj_wszystko(*dekoratory):
    def class_rebuilder(cls: type):
        for nazwa, wartosc in vars(cls).items():
            if callable(wartosc) and not nazwa.startswith("__"):
                for d in dekoratory:
                    wartosc = d(wartosc)
                setattr(cls, nazwa, wartosc)
        return cls

    return class_rebuilder
