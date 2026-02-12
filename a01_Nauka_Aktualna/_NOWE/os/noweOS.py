import os
from functools import wraps
from time import time


def czasFunkcji(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        katalogi, sprawdzone = func(*args, **kwargs)
        koniec = time()
        return katalogi, sprawdzone, koniec - start

    return wrapper


def zapisPlikow(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        wynik = func(*args, **kwargs)
        katalogi, _, _ = wynik

        if katalogi:
            with open("pamiecPlikow.txt", "a", encoding="utf-8") as plik:
                for sciezka in katalogi:
                    plik.write(f"{sciezka}\n")
        return wynik

    return wrapper


def przygotowanie(func):
    @wraps(func)
    def wrapper(lista_fraz, odKonca, zaczynaOd, maxWynikow, pamiec=None):
        maxWynikow += 1
        return func(lista_fraz, odKonca, zaczynaOd, maxWynikow, pamiec)

    return wrapper


def odbiorZapisu(func):
    @wraps(func)
    def wrapper(lista_fraz, odKonca, zaczynaOd, maxWynikow, pamiec=None):
        pamiec_z_pliku = []
        if os.path.exists("pamiecPlikow.txt"):
            with open("pamiecPlikow.txt", "r", encoding="utf-8") as plik:
                pamiec_z_pliku = [line.strip() for line in plik.readlines()]

        return func(lista_fraz, odKonca, zaczynaOd, maxWynikow, pamiec_z_pliku)

    return wrapper


@odbiorZapisu
@zapisPlikow
@czasFunkcji
@przygotowanie
def przeszukiwanie(
    lista_fraz=[], odKonca="", zaczynaOd="C:/", maxWynikow=100, pamiec=None
):
    katalogi = []
    sprawdzone = 0

    if isinstance(lista_fraz, str):
        lista_fraz = [lista_fraz]

    print(f"Rozpoczynam szukanie w: {zaczynaOd}")

    for root, _, files in os.walk(zaczynaOd):
        if sprawdzone % 10000 == 0 and sprawdzone > 0:
            print(f"Sprawdzono {sprawdzone} plików...")

        for plik in files:
            sprawdzone += 1
            if all(fraza.lower() in plik.lower() for fraza in lista_fraz):
                if odKonca == "" or plik.endswith(odKonca):
                    sciezka_pelna = os.path.join(root, plik)

                    if pamiec and sciezka_pelna in pamiec:
                        continue

                    katalogi.append(sciezka_pelna)
                    print(f"Znaleziono ({len(katalogi)}): {sciezka_pelna}")

                    if len(katalogi) >= maxWynikow:
                        return katalogi, sprawdzone

    return katalogi, sprawdzone


if __name__ == "__main__":

    wynik_koncowy = przeszukiwanie(
        lista_fraz=["git"], odKonca="", zaczynaOd="C:/", maxWynikow=100, pamiec=None
    )

    katalogi, sprawdzone, czas = wynik_koncowy

    print("\n" + "=" * 30)
    print(f"WYNIKI (Czas: {czas:.2f}s)")
    print(f"Sprawdzono plików: {sprawdzone}")
    for i, sciezka in enumerate(katalogi):
        print(f"{i+1}. {sciezka}")
