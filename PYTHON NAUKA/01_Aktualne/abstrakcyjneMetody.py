from abc import ABC, abstractmethod

class Edycja_(ABC):
    def __init__(self, szerokosc: int, wysokosc: int):
        self.szerokosc = szerokosc
        self.wysokosc = wysokosc
    @abstractmethod
    def zmienDane(self, szerokosc: int, wysokosc: int) -> None:
        pass
    @abstractmethod
    def zmienZapytanie(self, nowe_zapytanie: str) -> None:
        pass
    @abstractmethod
    def pokazZapytania(self) -> None:
        pass
    @abstractmethod
    def __str__(self) -> str:
        pass

class Edycja(Edycja_):
    def __init__(self, szerokosc: int, wysokosc: int, zapytanie: str):
        super().__init__(szerokosc, wysokosc)
        self.zapytania = {"szerokosc": self.szerokosc, "wysokosc": self.wysokosc}
        self.aktual_zapytanie = zapytanie
        self.iloscPrintow = 0

    def zmienDane(self, szerokosc, wysokosc):
        self.szerokosc = szerokosc
        self.wysokosc = wysokosc

        self.zapytania["szerokosc"] = szerokosc
        self.zapytania["wysokosc"] = wysokosc

    def zmienZapytanie(self, nowe_zapytanie):
        if nowe_zapytanie in self.zapytania:
            self.aktual_zapytanie = nowe_zapytanie
        else:
            print("Niepoprawne zapytanie!")

    def pokazZapytania(self) -> None:
        for i, (klucz, wartosc) in enumerate(self.zapytania.items()):
            print(f"{i}: zapytanie: {klucz}, wartość: {wartosc}")

    def __str__(self) -> str:
        self.iloscPrintow += 1
        return (
            f"{Edycja.__name__}, print nr.{self.iloscPrintow}: "
            f"szerokosc = {self.szerokosc}, wysokosc = {self.wysokosc}, "
            f"aktualne zapytanie = {self.aktual_zapytanie}"
        )

ob = Edycja(100, 200, "szerokosc")
print(ob)

ob.zmienDane(300, 200)
print(ob)

ob.zmienZapytanie("wysokosc")
ob.zmienDane(300, 400)
print(ob)

ob.pokazZapytania()
