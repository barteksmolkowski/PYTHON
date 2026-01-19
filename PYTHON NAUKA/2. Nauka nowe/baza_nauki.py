import os
import shutil
from functools import wraps
from typing import Optional

__all__ = ["__BazaNauki__", "bezpieczny_wrapper"]

class __BazaNauki__:
    def __init__(self, aktywne=True, dekoratory: Optional[list] = None):
        if not aktywne:
            print(f">>> SEKCJA POMINIĘTA: {self.__class__.__name__}")
            return
            
        print(f"\n>>> URUCHAMIAM SEKCJĘ: {self.__class__.__name__} <<<")
        for nazwa in dir(self):
            wartosc = getattr(self, nazwa)
            if callable(wartosc) and not nazwa.startswith("_") and nazwa != "run":
                wartosc()


class AutoDekoruj(type):
    """
    METAKLASA – KLASA, KTÓRA TWORZY INNE KLASY
    Python działa tak:
        obiekt  ← tworzony przez klasę
        klasa   ← tworzona przez metaklasę

    Ten __new__ uruchamia się W MOMENCIE TWORZENIA KLASY,
    a NIE podczas tworzenia jej instancji.

    Parametry __new__ metaklasy:

    cls:
        Metaklasa (np. AutoDekoruj), która buduje nową klasę.

    name:
        Nazwa nowo tworzonej klasy jako string.
        Przykład: "FileManager"

    bases:
        Krotka klas bazowych.
        Przykład: (__BazaNauki__,)

    metody_i_atrybuty (dct):
        Słownik zawierający CAŁE CIAŁO KLASY:
        - metody (funkcje)
        - atrybuty klasowe
        - dekoratory
        - zmienne

        Przykład:
        {
            'list_files': <function ...>,
            'create_directory': <function ...>,
            '__module__': 'file_manager',
            '__doc__': None
        }

    Dzięki metaklasie możemy:
        - automatycznie dekorować metody
        - modyfikować klasę bez pisania dekoratora na każdej metodzie
        - narzucać reguły całej klasie
    """
    def __new__(cls, name, base, dct):
        print(f"cls:\n{cls}")
        print(f"name:\n{name}")
        print(f"base:\n{base}")
        print(f"dct:\n{dct}")

        for k, v in dct.items():
            0

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
