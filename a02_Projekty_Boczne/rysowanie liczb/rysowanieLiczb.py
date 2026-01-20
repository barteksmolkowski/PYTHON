from abc import ABC, abstractmethod
from typing import Literal
import json
import pygame
import re

class __rysowanie__(ABC):
    def __init__(self, dane=None, macierz=None, nazwaPliku="rysowanie_zapis_plik"):
        super().__init__()
        pass

    @abstractmethod
    def __init__(self, dane=None, macierz=None, nazwaPliku="rysowanie_zapis_plik"):
        """
        Inicjalizuje obiekt rysowania.

        - Jeśli 'dane' nie zostaną podane, ustawiane są domyślne parametry rysunku
        (rozmiar siatki, wielkość pikseli, kolory, ramki).
        - Jeśli 'macierz' nie zostanie podana, tworzona jest nowa macierz 2D
        wypełniona kolorem początkowym.
        - 'nazwaPliku' określa domyślną nazwę pliku przy zapisie rysunku.
        """
        pass

    @abstractmethod
    def zmiana(self, el: str | list, nowy: int | list):
        """
        Zmienia wartości wybranych parametrów w słowniku self.dane.

        - 'el' może być nazwą jednego klucza lub listą kluczy.
        - 'nowy' może być jedną wartością lub listą nowych wartości.
        - Dla kluczy przechowujących kolor (tuple) wymagane są 3 wartości RGB.
        - Pozostałe wartości są rzutowane na int.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Zwraca czytelną, tekstową reprezentację obiektu.

        Każdy parametr z self.dane jest opisany:
        - nazwą po polsku,
        - jednostką (px, rgb lub brak),
        - sformatowaną wartością (RGB jako procenty).
        """
        pass

    @abstractmethod
    def kursos_na_kordy(self, x_Kursor: int, y_Kursor: int) -> None | tuple[int, int]:
        """
        Przelicza pozycję kursora (piksele ekranu) na współrzędne siatki (kolumna, wiersz).

        Zwraca:
        - (kolumna, wiersz) jeśli kursor znajduje się wewnątrz kwadratu siatki,
        - None jeśli kursor jest poza siatką, na ramce lub w przerwach między kwadratami.
        """
        pass

    @abstractmethod
    def dzialaj_na_pixel(self, co_zrob: Literal["zaznacz", "usun"], x: int, y: int) -> bool:
        """
        Zaznacza lub usuwa (resetuje) pojedynczy piksel w macierzy.

        - 'co_zrob' określa operację: "zaznacz" ustawia aktualny kolor,
        "usun" przywraca kolor początkowy.
        - Współrzędne (x, y) muszą mieścić się w granicach macierzy.
        - Zwraca True, jeśli operacja się powiodła, w przeciwnym razie False.
        """
        pass

    @abstractmethod
    def odczytaj(self, sciezka_nazwa):
        if not sciezka_nazwa.endswith(".json"):
            sciezka_nazwa += ".json"

        with open(sciezka_nazwa, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "dane" not in data or "macierz" not in data:
            raise ValueError("Niepoprawny plik zapisu")

        self.dane = data["dane"]
        self.macierz = data["macierz"]

    @abstractmethod
    def zapisz(self, nazwa=None):
        if nazwa is None:
            nazwa = self.nazwaPliku

        with open(f"{nazwa}.json", "w", encoding="utf-8") as f:
            json.dump(
                {"dane": self.dane, "macierz": self.macierz},
                f,
                indent=4
            )

class rysowanie(__rysowanie__):
    def __init__(self, dane=None, macierz=None, nazwaPliku="rysowanie_zapis_plik"):
        super().__init__(dane, macierz, nazwaPliku)
        if dane is None:
            self.dane = {
                "szerIloscKwad": 20,
                "wysIloscKwad": 4,
                "szerPx": 5,
                "wysPx": 5,
                "szerRamka": 7,
                "wysRamka": 8,
                "odzielnieSzer": 9,
                "odzielnieWys": 1,
                "kolorRamki": (255, 253, 240),
                "pierwszyKolorPx": (211, 211, 211),
                "aktualKolor": (0, 0, 0)
            }
        else:
            self.dane = dane

        if macierz is None:
            self.macierz = [
                [self.dane["pierwszyKolorPx"]
                for _ in range(self.dane["szerIloscKwad"])]
                for _ in range(self.dane["wysIloscKwad"])
            ]
        else:
            self.macierz = macierz

        self.nazwaPliku = nazwaPliku

    def zmiana(self, el: str | list, nowy: int | list):
        el = [el] if isinstance(el, str) else el
        nowy = [nowy] if not isinstance(nowy, list) else nowy

        if len(el) != len(nowy):
            raise ValueError("Listy el i nowy muszą mieć tę samą długość")

        for k, v in zip(el, nowy):
            if k not in self.dane:
                raise KeyError(f"Nie ma klucza: {k}")

            if isinstance(self.dane[k], tuple):
                if not (isinstance(v, (list, tuple)) and len(v) == 3):
                    raise ValueError(f"{k} musi być RGB (3 wartości)")
                self.dane[k] = tuple(v)
            else:
                self.dane[k] = int(v)


    def __str__(self): # dokończyć
        # dopełniać 0 do % rgb
        # zrobić najlepiej ramkę gdzie poprostu wszystko będzie
        slow = {
                "szerIloscKwad": ["Ilość kwadratów w szerokości", ""],
                "wysIloscKwad": ["Ilość kwadratów w wysokości", ""],
                "szerPx": ["Szerokość pixela", "px"],
                "wysPx": ["Wysokość pixela", "px"],
                "szerRamka": ["Szerokość ramki", "px"],
                "wysRamka": ["Wysokość ramki", "px"],
                "odzielnieSzer": ["Szerokość przerwy", "px"],
                "odzielnieWys": ["Wysokość przerwy", "px"],
                "kolorRamki": ["Kolor ramki", "rgb"],
                "pierwszyKolorPx": ["Początkowy kolor okienka", "rgb"],
                "aktualKolor": ["Aktualny kolor", "rgb"]
                }
        
        wynik = ""

        for i, (nazwa, el) in enumerate(self.dane.items()):
            rodzaj = slow[nazwa][1]

            if rodzaj in ["", "px"]:
                koniec = f"{el}{slow[nazwa][1]}"

            elif rodzaj in "rgb":
                koniec = (
                    f"R:{el[0]/255:.0%} "
                    f"G:{el[1]/255:.0%} "
                    f"B:{el[2]/255:.0%}")

            wynik += f"el{i + 1}. {slow[nazwa][0]}: {koniec}\n"

        return wynik

    def kursos_na_kordy(self, x_Kursor, y_Kursor):
        d = self.dane

        x = x_Kursor - d["szerRamka"]
        y = y_Kursor - d["wysRamka"]

        if x < 0 or y < 0:
            return None

        blok_szer = d["szerPx"] + d["odzielnieSzer"]
        blok_wys  = d["wysPx"]  + d["odzielnieWys"]

        kolumna = x // blok_szer
        wiersz  = y // blok_wys

        x_w_bloku = x % blok_szer
        y_w_bloku = y % blok_wys

        if x_w_bloku >= d["szerPx"]:
            return None
        if y_w_bloku >= d["wysPx"]:
            return None

        if (
            kolumna < 0 or kolumna >= d["szerIloscKwad"] or
            wiersz  < 0 or wiersz  >= d["wysIloscKwad"]
        ):
            return None

        return (kolumna, wiersz)

    def dzialaj_na_pixel(self, co_zrob: Literal["zaznacz", "usun"], x: int, y: int) -> bool:
        if 0 <= x < self.dane["szerIloscKwad"] and 0 <= y < self.dane["wysIloscKwad"]:
            self.macierz[y][x] = self.dane["aktualKolor"] if co_zrob == "zaznacz" else self.dane["pierwszyKolorPx"]
        else:
            return False

    def odczytaj(self, sciezka_nazwa):
        if not sciezka_nazwa.endswith(".json"):
            sciezka_nazwa += ".json"

        with open(sciezka_nazwa, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "dane" not in data or "macierz" not in data:
            raise ValueError("Niepoprawny plik zapisu")

        self.dane = data["dane"]
        self.macierz = data["macierz"]

    def zapisz(self, nazwa=None):
        if nazwa is None:
            nazwa = self.nazwaPliku

        with open(f"{nazwa}.json", "w", encoding="utf-8") as f:
            json.dump(
                {"dane": self.dane, "macierz": self.macierz},
                f,
                indent=4
            )

class rysuj:
    def rysuj_tlo(self, screen):
        screen.fill((30, 30, 30))

    def rysuj_ramke(self, screen, dane, szer, wys):
        pygame.draw.rect(
            screen,
            dane["kolorRamki"],
            (0, 0, szer, wys),
            dane["szerRamka"]
        )

    def rysuj_pixele(self, screen, dane, macierz):
        for y in range(dane["wysIloscKwad"]):
            for x in range(dane["szerIloscKwad"]):
                kolor = macierz[y][x]

                px = dane["szerRamka"] + x * (dane["szerPx"] + dane["odzielnieSzer"])
                py = dane["wysRamka"]  + y * (dane["wysPx"]  + dane["odzielnieWys"])

                pygame.draw.rect(
                    screen,
                    kolor,
                    (px, py, dane["szerPx"], dane["wysPx"])
                )

    def rysuj_wszystko(self, screen, dane, macierz, kordy, szer, wys):
        self.rysuj_tlo(screen)
        self.rysuj_ramke(screen, dane, szer, wys)
        self.rysuj_pixele(screen, dane, macierz)

class ekran:
    def __init__(self, dane):
        self.model = rysowanie(dane)
        self.view = rysuj()

    def obsluga_przyciskow(self, event):
        eventType = event.type
        eventKey = event.key
        kolory = {
            pygame.K_1: (255, 0, 0),      # czerwony
            pygame.K_2: (0, 255, 0),      # zielony
            pygame.K_3: (0, 0, 255),      # niebieski
            pygame.K_4: (255, 255, 0),    # żółty
            pygame.K_5: (255, 165, 0),    # pomarańczowy
            pygame.K_6: (128, 0, 128),    # fioletowy
            pygame.K_7: (0, 255, 255),    # cyjan
            pygame.K_8: (255, 192, 203),  # różowy
            pygame.K_9: (128, 128, 128),  # szary
        }
        if eventType == pygame.QUIT:
            self.wlaczone = False

        elif eventType == pygame.KEYDOWN:
            k = eventKey

            if self.wpisywanie_liczb:
                if k == pygame.K_p:

                    if self.aktualna_liczba:
                        val = int(self.aktualna_liczba)

                        if 0 <= val <= 255:
                            self.liczby_aktualne.append(val)
                            self.aktualna_liczba = ""
                        else:

                            self.aktualna_liczba = self.aktualna_liczba[:-1]

                    if len(self.liczby_aktualne) == 3:
                        self.model.dane["aktualKolor"] = tuple(self.liczby_aktualne)
                        self.wpisywanie_liczb = False
                        self.liczby_aktualne = []
                        self.aktualna_liczba = ""
                    return

                elif k == pygame.K_r:
                    self.aktualna_liczba = ""
                    return

                elif pygame.K_0 <= k <= pygame.K_9:
                    if len(self.aktualna_liczba) < 3:
                        self.aktualna_liczba += chr(k)
                    return

            elif self.wpisywanie_nazw:
                if k == pygame.K_RETURN:

                    kolor = self.dopasuj_nazwe_koloru(self.nazwa_aktualna)
                    if kolor:
                        self.model.dane["aktualKolor"] = kolor
                    self.wpisywanie_nazw = False
                    self.nazwa_aktualna = ""
                    return
                elif k == pygame.K_BACKSPACE:
                    self.nazwa_aktualna = self.nazwa_aktualna[:-1]
                    return
                else:
                    znak = event.unicode
                    if znak.isalpha():
                        self.nazwa_aktualna += znak
                    return

            else:
                if k in kolory:
                    self.model.dane["aktualKolor"] = kolory[k]
                    return
                if k == pygame.K_p:

                    self.wpisywanie_liczb = True
                    self.liczby_aktualne = []
                    self.aktualna_liczba = ""
                    return
                if k == pygame.K_l:

                    self.wpisywanie_nazw = True
                    self.nazwa_aktualna = ""
                    return

    def dopasuj_nazwe_koloru(self, tekst):
        nazwy = {
            "czerwony": (255,0,0),
            "zielony": (0,255,0),
            "niebieski": (0,0,255),
            "zolty": (255,255,0),
            "pomaranczowy": (255,165,0),
            "fioletowy": (128,0,128),
            "cyjan": (0,255,255),
            "rozowy": (255,192,203),
            "szary": (128,128,128)
        }
        for nazwa, kolor in nazwy.items():
            if re.search(tekst.lower(), nazwa):
                return kolor
        return None

    def uruchom(self):
        pygame.init()
        d = self.model.dane

        szer = (
            d["szerIloscKwad"] * d["szerPx"]
            + (d["szerIloscKwad"] - 1) * d["odzielnieSzer"]
            + 2 * d["szerRamka"]
        )

        wys = (
            d["wysIloscKwad"] * d["wysPx"]
            + (d["wysIloscKwad"] - 1) * d["odzielnieWys"]
            + 2 * d["wysRamka"]
        )

        screen = pygame.display.set_mode((szer, wys))
        zegar = pygame.time.Clock()

        self.wlaczone = True

        while self.wlaczone:
            zegar.tick(60)

            for event in pygame.event.get():
                self.obsluga_przyciskow(event)

            mx, my = pygame.mouse.get_pos()
            kordy = self.model.kursos_na_kordy(mx, my)

            if pygame.mouse.get_pressed()[0] and kordy:
                if kordy:
                    self.model.dzialaj_na_pixel("zaznacz", *kordy)

            self.view.rysuj_wszystko(
                screen,
                d,
                self.model.macierz,
                kordy,
                szer,
                wys
            )

            pygame.display.flip()

        pygame.quit()


rys = rysowanie()
rys.uruchom()
