from abc import ABC, abstractmethod

class Pobierz_tresc(ABC):
    def __init__(self, wartosc, numer_dnia):
        self.wartosc = wartosc
        self.linie_w_plik = []

    @abstractmethod
    def zapisz_wartosc(self):
        pass
    
    @abstractmethod
    def odczytaj_wartosc(self):
        pass