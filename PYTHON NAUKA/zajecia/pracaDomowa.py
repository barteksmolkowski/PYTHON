# random można z czasem zrobić i mnożyć milisekundy (jakąś część) razy wielkość tabeli i modulo 10 aby 0-9 liczby
import math
from datetime import datetime
import time

class Losowanie:
    def __init__(self, od, do, ile):
        self.od = od
        self.do = do
        self.ile = ile
        self.wyniki = self.losuj()

    def __str__(self):
        return ", ".join(map(str, self.wyniki))

    def __wygenerujJedna(self):
        czas = datetime.now()

        mikro = czas.microsecond

        cyfra = mikro % 10
        if cyfra == 0:
            cyfra = 7

        liczba = mikro + (time.time_ns() % 100000)

        lista = list(str(liczba).rstrip("0"))
        idx = cyfra % len(lista)
        wynik = int(lista[idx])

        if wynik == 0:
            wynik = 9

        return wynik

    def __random(self, ile, startowa):
        wynik = []
        liczba_glowna = startowa
        rejestr_pomocniczy = int(time.time_ns()) & 0xFFFFFFFF

        for _ in range(ile):
            while True:
                wygenerowane = []

                for _ in range(10):
                    liczba_glowna = (liczba_glowna * 7397510893 + rejestr_pomocniczy) % (2**32)
                    rejestr_pomocniczy = (rejestr_pomocniczy * 1664525 + 1013904223) % (2**32)
                    wartosc = (liczba_glowna ^ rejestr_pomocniczy) & 0xF
                    wygenerowane.append(wartosc)

                dobre = [x for x in wygenerowane if x in [5,6,7,8]]

                if dobre:
                    wybrana = dobre[0]
                    bit = 0 if wybrana in [5,8] else 1
                    wynik.append(bit)
                    break
        return wynik

    def losuj(self):
        wyniki = []
        print("losowanie...")

        for _ in range(self.ile):
            time.sleep(0.05)

            low = self.od
            high = self.do
            zakres = high - low
            ile_bitow = math.ceil(math.log2(zakres + 1))

            startowa = self.__wygenerujJedna()
            liczby = self.__random(ile_bitow, startowa)

            for bit in liczby:
                if low == high:
                    break
                mid = (low + high) // 2
                if bit == 0:
                    high = mid
                else:
                    low = mid + 1

            wyniki.append(low)

        return wyniki

# los = Losowanie(1, 10, 300)

# slownik = {}
# for x in los.wyniki:
#     slownik[x] = slownik.get(x, 0) + 1

# for k, v in sorted(slownik.items()):
#     print(f"Liczbę{k} wylosowano razy {v}.")

# Stwórz liste zawierającą 100 randomowo generowanych elementow w zakaresie od 1-20
# jezeli chętny mozesz zaimplementwoać generator liczb pseudolosowych
# Celem zadania jest znalezienie sumy liczb o nieokreślonej ilości, których wartośc jest równa 
# target,
# 1. że znajdź pierwszą możliwą kombinacje, nastepnie wypisz indeksy tych elementów
# 2. znajdź liczbę kombinacji których suma jest równa targetowi
# 3. porządek leksykograficzny
# przykład 1st:1 (1-pozycja na liście), 2nd:2, 3rd: 3 => sumarycznie równe
# 015 123 312 <- to porządek (bez powtórek, tu np 312 musi zniknąć bo jest 123)
# import random

import random

class KombinacjeSum:
    def __init__(self, od_, do_, ilosc, target, max_wynikow):
        self.od = od_
        self.do = do_
        self.ilosc = ilosc
        self.target = target
        self.max_wynikow = max_wynikow
        self.losowe = []
        self.wynik = []
    
    def __str__(self):
        self.generuj_losowe()
        self.run()
        return f"{self.wynik}"

    def generuj_losowe(self):
        self.losowe = [random.randint(self.od, self.do) for _ in range(self.ilosc)]
        print("Wylosowana lista:", self.losowe)

    def kombinacje_dlugosci_k(self, ilosc_elementow, dlugosc):
        kombinacja = list(range(dlugosc))

        while True:

            yield kombinacja[:]
            i = dlugosc - 1

            while i >= 0 and kombinacja[i] == ilosc_elementow - (dlugosc - i):
                i -= 1

            if i < 0:
                break

            kombinacja[i] += 1

            for j in range(i + 1, dlugosc):
                kombinacja[j] = kombinacja[j - 1] + 1

    def znajdz_kombinacje(self, target, max_wynikow):
        n = len(self.losowe)
        znalezione = 0

        for dlugosc in range(1, n + 1):

            for el in self.kombinacje_dlugosci_k(n, dlugosc):

                if sum(self.losowe[i] for i in el) == target:
                    nowa_lista = tuple((i, self.losowe[i]) for i in el)
                    yield nowa_lista
                    znalezione += 1
                    if znalezione >= max_wynikow:
                        return
                    
            if n % 100 == 0:
                print(f"Sprawdzono {n} możliwości.")

    def run(self):
        if not self.losowe:
            self.generuj_losowe()

        print(f"Target: {self.target}")
        licznik = 0

        for wynik in self.znajdz_kombinacje(self.target, self.max_wynikow):
            licznik += 1

            indeksy = tuple(idx for idx, _ in wynik)
            indeksy = indeksy[0] if len(indeksy) == 1 else indeksy
            wartosci = [val for _, val in wynik]
            suma_str = " + ".join(str(v) for v in wartosci)
            suma = sum(wartosci)

            if "+" in suma_str:
                print(f"nr.{licznik}: indeksy: {indeksy} suma: {suma_str} = {suma}")
            else:
                print(f"nr.{licznik}: indeksy: {indeksy} suma: {suma_str}")
            self.wynik.append(indeksy)

        print(f"\nLiczba znalezionych kombinacji: {licznik}")

kombinacje_sum = KombinacjeSum(od_=1, do_=20, ilosc=100, target=20, max_wynikow=100)
print(f"Ostateczne wyniki: {kombinacje_sum}")

# (0, 0, 0)