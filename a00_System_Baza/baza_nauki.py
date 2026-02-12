import os
import shutil
import sys
from functools import wraps

__all__ = ["__BazaNauki__", "bezpieczny_wrapper", "dekoruj_wszystko"]

import msvcrt
import re
from typing import Optional


class __BazaNauki__:
    def __init__(
        self,
        aktywne=True,
        pokaz_docstring: bool = True,
        czekaj_na_enter: bool = False,
        czysc_ekran: bool = False,
        interaktywne: bool = False,
        tylko_dane: bool = False,
        lista_klas: Optional[list[type]] = None,
        wybrane_metody_f: Optional[list[str]] = None,
    ):
        self.edukator = Edukator
        self.lista_klas = lista_klas

        if not aktywne:
            return

        wszystkie_publiczne = [
            n for n in dir(self)
            if callable(getattr(self, n))
            and not n.startswith("_")
            and n not in ["run", "Edukator"]
            and n in dir(self.__class__)
        ]

        kolejnosc_z_opisu = []
        if self.__doc__:
            for line in self.__doc__.strip().split("\n"):
                if " - " in line:
                    part = line.split(" - ")[0]
                    clean_name = part.replace("│", "").replace("║", "").strip().split()[-1]
                    kolejnosc_z_opisu.append(clean_name)

        if wybrane_metody_f:
            filtr_clean = [s.strip() for s in wybrane_metody_f]
            
            self.wybrane_metody = [n for n in filtr_clean if n in wszystkie_publiczne]
        else:
            finalna_lista = [n for n in kolejnosc_z_opisu if n in wszystkie_publiczne]
            reszta = sorted([n for n in wszystkie_publiczne if n not in finalna_lista])
            self.wybrane_metody = finalna_lista + reszta


        if tylko_dane:
            return

        if interaktywne:
            self._silnik_interaktywny()
        else:
            self._silnik_standardowy(pokaz_docstring, czysc_ekran, czekaj_na_enter)

    def _silnik_interaktywny(self):
        if self.lista_klas:
            self.edukator.start(self.lista_klas)
        else:
            self.edukator.start([self.__class__])

    def _rysuj_opis(self, aktywny_idx: Optional[int] = None):
        if self.__doc__:
            lines = [l.strip() for l in self.__doc__.strip().split("\n") if l.strip()]
            print("┌" + "─" * 10 + " OPIS SEKCJI " + "─" * 38 + "\033[K")
            for idx, line in enumerate(lines):
                if idx == aktywny_idx:
                    print(f"│ \033[1;32m{idx+1:2} ║ {line}\033[0m\033[K")
                else:
                    print(f"│ {idx+1:2} ║ {line}\033[K")
            print("└" + "─" * 56 + "\033[K")

    def _silnik_standardowy(self, pokaz_docstring, czysc_ekran, czekaj_na_enter):
        if czysc_ekran:
            os.system("cls" if os.name == "nt" else "clear")
        if pokaz_docstring:
            self._rysuj_opis()
        for nazwa in self.wybrane_metody:
            getattr(self, nazwa)()
            if czekaj_na_enter:
                input("\n[ENTER -> Kolejny]")


class Edukator:
    @staticmethod
    def nawiguj(
        tytul: str, opcje: list, doc_rysuj_func=None, start_idx: int = 0
    ) -> int:
        idx = start_idx
        pierwszy_raz = True

        wysokosc_opisu = 0
        if (
            doc_rysuj_func
            and hasattr(doc_rysuj_func, "__self__")
            and doc_rysuj_func.__self__.__doc__
        ):
            wysokosc_opisu = (
                len(doc_rysuj_func.__self__.__doc__.strip().split("\n")) + 2
            )

        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

        try:
            while True:
                if pierwszy_raz:
                    os.system("cls" if os.name == "nt" else "clear")
                    print(f"\033[1;34m╔{'═' * (len(tytul) + 2)}╗\033[0m")
                    print(f"\033[1;34m║ {tytul} ║\033[0m")
                    print(f"\033[1;34m╚{'═' * (len(tytul) + 2)}╝\033[0m")
                    print(
                        "\033[90m[↑/↓: Nawigacja | Enter: Wybierz | Q/Esc: Wyjdź]\033[0m"
                    )
                    print()
                    if doc_rysuj_func:
                        for _ in range(wysokosc_opisu + 1):
                            print()
                    for _ in range(len(opcje)):
                        print()
                    pierwszy_raz = False

                if doc_rysuj_func:
                    wysokosc_powrotu = wysokosc_opisu + len(opcje) + 1
                else:
                    wysokosc_powrotu = len(opcje)

                sys.stdout.write(f"\r\033[{wysokosc_powrotu}A")

                if doc_rysuj_func:
                    doc_rysuj_func(aktywny_idx=idx)
                    print("\033[K")

                for i, opcja in enumerate(opcje):
                    if i == idx:
                        sys.stdout.write(
                            f"\r \033[1;30;42m {i+1:2} ║ {opcja.ljust(60)} \033[0m\033[K\n"
                        )
                    else:
                        sys.stdout.write(f"\r    {i+1:2} ║ {opcja.ljust(60)}\033[K\n")

                sys.stdout.write("\033[J")
                sys.stdout.flush()

                klawisz = msvcrt.getch()
                if klawisz == b"\xe0":
                    kierunek = msvcrt.getch()
                    if kierunek == b"H" and idx > 0:
                        idx -= 1
                    elif kierunek == b"P" and idx < len(opcje) - 1:
                        idx += 1
                elif klawisz in [b"\r", b"\n"]:
                    return idx
                elif klawisz.lower() == b"q" or klawisz == b"\x1b":
                    return -1
        finally:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

    @classmethod
    def start(cls, lista_klas: list):
        idx_glowne = 0
        pamiec_pozycji = {k.__name__: 0 for k in lista_klas}

        while True:
            menu_wyswietlane = []
            mapa_danych = [] 

            MAX_OPIS = 50 

            for k in lista_klas:
                pelny_opis = getattr(k, 'opis_menu', "brak opisu")
                slowa = [s.strip() for s in pelny_opis.split(",")]
                
                czesci_slow = []
                aktualna = []
                for s in slowa:
                    if not s: continue
                    if len(", ".join(aktualna + [s])) <= MAX_OPIS:
                        aktualna.append(s)
                    else:
                        czesci_slow.append(aktualna)
                        aktualna = [s]
                if aktualna: czesci_slow.append(aktualna)

                for i, grupa in enumerate(czesci_slow):
                    licznik = f" {i+1}/{len(czesci_slow)}" if len(czesci_slow) > 1 else ""
                    nazwa_menu = (k.__name__ + licznik).ljust(20)
                    tekst_opisu = ", ".join(grupa)
                    
                    menu_wyswietlane.append(f"{nazwa_menu} # {tekst_opisu}")
                    mapa_danych.append((k, grupa))

            wybrany_idx = cls.nawiguj("BIBLIOTEKA WIEDZY", menu_wyswietlane, start_idx=idx_glowne)
            if wybrany_idx == -1: break
            
            idx_glowne = wybrany_idx
            klasa_obj, filtr_metod = mapa_danych[wybrany_idx]
            
            instancja = klasa_obj(aktywne=True, tylko_dane=True, wybrane_metody_f=filtr_metod)
            nazwa_r = klasa_obj.__name__

            while True:
                wybrany_metoda_idx = cls.nawiguj(
                    f"ROZDZIAŁ: {nazwa_r}",
                    instancja.wybrane_metody,
                    doc_rysuj_func=instancja._rysuj_opis,
                    start_idx=pamiec_pozycji[nazwa_r],
                )

                if wybrany_metoda_idx == -1:
                    break

                pamiec_pozycji[nazwa_r] = wybrany_metoda_idx

                os.system("cls")
                metoda_nazwa = instancja.wybrane_metody[wybrany_metoda_idx]
                print(f"\033[1;33m>>> URUCHOMIONO: {metoda_nazwa} <<<\033[0m\n")

                import builtins
                stary_print = builtins.print
                builtins.print = cls.koloruj_tekst
                try:
                    getattr(instancja, metoda_nazwa)()
                finally:
                    builtins.print = stary_print

                print("\n\033[90m" + "─" * 50 + "\n[Naciśnij dowolny klawisz, aby wrócić]\033[0m")
                msvcrt.getch()


    @staticmethod
    def koloruj_tekst(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        tekst = sep.join(map(str, args))

        keywords = r"\b(git|python|pip|cd|ls|mkdir|rm|cp|mv|def|class|import|from|if|elif|else|return|try|except|with|as|for|while|in|is|not|and|or|lambda|None|True|False|switch|restore|checkout|commit|push|pull|status|add|log|branch|reset|soft|hard|diff|merge|rebase|bool|int|str|float|list|dict|set)\b"

        for linia in tekst.splitlines():
            l = linia

            if l.strip() == "---":
                sys.stdout.write("\033[90m" + "─" * 50 + "\033[0m\n")
                continue

            komentarz = ""
            if "#" in l:
                czesci = l.split("#", 1)
                l = czesci[0]
                komentarz = f"\033[90m#{czesci[1]}\033[0m"

            l = re.sub(r'("[^"]*"|\'[^\']*\')', r"\033[1;32m\1\033[0m", l)
            l = re.sub(r"\b(ERROR|BŁĄD|FAIL)\b", r"\033[1;41;37m \1 \033[0m", l)
            l = re.sub(r"\b(WARNING|UWAGA|INFO)\b", r"\033[1;43;30m \1 \033[0m", l)
            l = re.sub(r"(@[a-zA-Z_]\w*)", r"\033[1;35m\1\033[0m", l)
            l = re.sub(r"\b([a-zA-Z_]\w*)\s*(?=\()", r"\033[1;36m\1\033[0m", l)
            l = re.sub(f"(?<!\033\[){keywords}", r"\033[1;33m\1\033[0m", l)
            l = re.sub(r"(?<![\[;0-9])\b(\d+)\b", r"\033[1;35m\1\033[0m", l)
            l = re.sub(r"(^|\s)(-[a-zA-Z0-9-]+)", r"\1\033[36m\2\033[0m", l)
            l = re.sub(r"(<[^>]+>)", r"\033[92m\1\033[0m", l)
            l = re.sub(r"(https?://\S+)", r"\033[4;34m\1\033[0m", l)
            l = l.replace("->", "\033[1;97m->\033[0m")
            l = l.replace("$ ", "\033[1;97m$ \033[0m")

            sys.stdout.write(l + komentarz + "\033[0m\n")

        if end != "\n":
            sys.stdout.write(end)
            sys.stdout.flush()


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
