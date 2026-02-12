from typing import Protocol

from a00_System_Baza.baza_nauki import __BazaNauki__


class MetaprogramowaniePodstawy(Protocol):
    def roznica_vars_dict(self): ...
    def problem_modyfikacji_slownika(self): ...
    def specyfika_classmethod(self): ...
    def pokaz_kod_automatu(self): ...


class LekcjaBebechyPythona(MetaprogramowaniePodstawy, __BazaNauki__):
    def pokaz_kod_automatu(self) -> None:
        print("=== WZORZEC PANCERNEGO DEKORATORA KLASY (2026) ===")
        print(
            f"def dekoruj_wszystko(*dekoratory):\n"
            f"{' '*4}def class_rebuilder(cls: type):\n"
            f"{' '*8}# Zamiana na list() zapobiega błędowi 'dictionary changed size'\n"
            f"{' '*8}for nazwa, wartosc in list(vars(cls).items()):\n"
            f"{' '*12}# Sprawdzamy czy to metoda (callable) lub @classmethod\n"
            f"{' '*12}if (callable(wartosc) or isinstance(wartosc, classmethod)) and not nazwa.startswith('__'):\n"
            f"{' '*16}for d in dekoratory:\n"
            f"{' '*20}# Jeśli to classmethod, dekorujemy funkcję w środku (.__func__)\n"
            f"{' '*20}if isinstance(wartosc, classmethod):\n"
            f"{' '*24}wartosc = classmethod(d(wartosc.__func__))\n"
            f"{' '*20}else:\n"
            f"{' '*24}wartosc = d(wartosc)\n"
            f"{' '*16}setattr(cls, nazwa, wartosc)\n"
            f"{' '*8}return cls\n"
            f"{' '*4}return class_rebuilder\n"
        )

    def roznica_vars_dict(self) -> None:
        print("VARS(CLS) VS CLS.__DICT__")
        print("vars(cls) to oficjalny interfejs Pythona (czysty i czytelny).")
        print("cls.__dict__ to surowy dostęp do bebechów klasy.")
        print(
            "Obie metody pokazują to samo, ale vars() jest bezpieczniejszym standardem w 2026 roku."
        )
        print(
            "Pamiętaj: w obu zobaczysz systemowe klucze (__module__, __doc__), które trzeba odfiltrować.\n"
        )

    def problem_modyfikacji_slownika(self) -> None:
        print("PROBLEM Z LIST(VARS(CLS).ITEMS())")
        print(
            "Próba zmiany klasy (setattr) w trakcie pętli po vars(cls).items() wywoła błąd."
        )
        print(
            "Python nie pozwala zmieniać rozmiaru słownika, który aktualnie przeglądasz."
        )
        print("Rozwiązanie: list(...) tworzy kopię (snapshot) kluczy.")
        print(
            "Dzięki temu możesz bezpiecznie dekorować metody 'w locie' bez zatrzymania programu.\n"
        )

    def specyfika_classmethod(self) -> None:
        print("DLACZEGO CLASSMETHOD JEST SPECJALNY W VARS(CLS)?")
        print(
            "Metoda z @classmethod nie jest zwykłą funkcją (callable) w słowniku klasy."
        )
        print("W vars(cls) widnieje jako obiekt typu 'classmethod' (deskryptor).")
        print("Aby ją udekorować, musisz sięgnąć głębiej do atrybutu .__func__.")
        print(
            "Po udekorowaniu musisz ją ponownie opakować w classmethod(), inaczej straci dostęp do 'cls'.\n"
        )


if __name__ == "__main__":
    LekcjaBebechyPythona(True)
