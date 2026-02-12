from typing import Protocol

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from a00_System_Baza.baza_nauki import __BazaNauki__


class WalidacjaIDetekcja_Protocol(Protocol):
    """
    SEKCJA LOGIKI:
    weryfikacja_logiczna_all              -> np.all
    weryfikacja_logiczna_any              -> np.any
    detekcja_brakow_isnan                 -> np.isnan
    porownanie_bliskosci_allclose         -> np.allclose
    operacje_porownania_greater           -> np.greater
    """

    def weryfikacja_logiczna_all(self) -> None: ...
    def weryfikacja_logiczna_any(self) -> None: ...
    def detekcja_brakow_isnan(self) -> None: ...
    def porownanie_bliskosci_allclose(self) -> None: ...
    def operacje_logiczne_greater(self) -> None: ...
class GeneratorStaly_Protocol(Protocol):
    """
    GENEROWANIE DANYCH PRZEWIDYWALNYCH:
    - inicjalizacja_zeros_ones_full  -> np.zeros, np.ones, np.full
    - sekwencje_arange_reshape_eye -> np.arange, np.reshape, np.eye
    - podzial_linspace             -> np.linspace
    """

    def inicjalizacja_zeros_ones_full(self) -> None: ...
    def sekwencje_arange_reshape_eye(self) -> None: ...
    def podzial_linspace(self) -> None: ...
class GeneratorLosowy_Protocol(Protocol):
    """
    GENEROWANIE DANYCH NIEPRZEWIDYWALNYCH I SPECJALNYCH:
    - rozklady_rand_normal         -> np.random.rand, np.random.normal
    - losowosc_default_rng_choice  -> np.random.default_rng, rng.choice
    - struktury_diagonalne_diag    -> np.diag
    """

    def rozklady_rand_normal(self) -> None: ...
    def losowosc_default_rng_choice(self) -> None: ...
    def struktury_diagonalne_diag(self) -> None: ...
class MagazynDanych_Protocol(Protocol):
    """
    SEKCJA PLIKÓW:
    trwalosc_save_load_savetxt            -> np.save, np.load, np.savetxt, np.loadtxt
    """

    def trwalosc_save_load_savetxt(self) -> None: ...
class MacierzowaGeometria_Protocol(Protocol):
    """
    SEKCJA STRUKTURY:
    otaczanie_marginesem_pad              -> np.pad
    zmiana_ukladu_reshape_view            -> np.reshape (zaawansowany)
    okna_przesuwne_sliding_window_view    -> np.lib.stride_tricks.sliding_window_view
    zarzadzanie_pamiecia_copy             -> np.copy
    """

    def otaczanie_marginesem_pad(self) -> None: ...
    def zmiana_ukladu_reshape_view(self) -> None: ...
    def okna_przesuwne_sliding_window_view(self) -> None: ...
    def ekstrakcja_sliding_window_view(self) -> None: ...
    def zarzadzanie_pamiecia_copy(self) -> None: ...
class ProcesorAlgorytmow_Protocol(Protocol):
    """
    SEKCJA ALGORYTMÓW:
    ekstrakcja_sliding_window_view        -> np.lib.stride_tricks.sliding_window_view
    mnozenie_macierzowe_einsum            -> np.einsum
    pooling_window_plus_einsum            -> sliding_window_view + einsum
    """

    def mnozenie_macierzowe_einsum(self) -> None: ...
    def pooling_window_plus_einsum(self) -> None: ...
class AnalizaIZbiory_Protocol(Protocol):
    """
    INSPEKCJA WARTOŚCI I OPERACJE ZBIOROWE:
    - detekcja_maksimow_argmax     -> np.argmax (pozycja największej wartości)
    - unikalnosc_unique_all        -> np.unique (wyciąganie unikalnych elementów/kolumn)
    - czesci_wspolne_intersect1d   -> np.intersect1d (szukanie wspólnych wartości)
    - laczenie_macierzy_append     -> np.append (rozbudowa zbioru danych)
    """

    def detekcja_maksimow_argmax(self) -> None: ...
    def unikalnosc_unique_all(self) -> None: ...
    def czesci_wspolne_intersect1d(self) -> None: ...
    def laczenie_macierzy_append(self) -> None: ...


class WalidacjaIDetekcja(__BazaNauki__, WalidacjaIDetekcja_Protocol):
    opis_menu = "weryfikacja_logiczna_all, weryfikacja_logiczna_any, detekcja_brakow_isnan, porownanie_bliskosci_allclose, operacje_logiczne_greater"
    
    """
    weryfikacja_logiczna_all - sprawdzanie czy wszystkie elementy spełniają warunek,
    weryfikacja_logiczna_any - sprawdzanie czy jakikolwiek element spełnia warunek,
    detekcja_brakow_isnan - identyfikacja wartości nieokreślonych NaN,
    porownanie_bliskosci_allclose - bezpieczne porównywanie float z tolerancją,
    operacje_logiczne_greater - wektorowe operacje porównania i maski
    """

    def weryfikacja_logiczna_all(self):
        print("INFO: WERYFIKACJA LOGICZNA (np.all)")
        print("-> A = np.array([[3, 2, 1, 4], [5, 2, 1, 6]])")
        print("---")
        print("Czy wszystkie elementy w rzędach (axis=1) są True / niezerowe?")
        print("-> wynik = np.all(A, axis=1)  # Logiczne AND")
        print("---")
        print("Wynik walidacji:")
        print("-> [True True]  # Wszystkie elementy są > 0")
        print("INFO: Kluczowe przy walidacji batchy danych przed wejściem do modelu.")

    def weryfikacja_logiczna_any(self):
        print("INFO: WERYFIKACJA LOGICZNA (np.any)")
        print("-> B = np.array([[0, 0, 0], [0, 1, 0]])")
        print("---")
        print("Czy w kolumnach (axis=0) występuje choć jedna wartość True / != 0?")
        print("-> wynik = np.any(B, axis=0)  # Logiczne OR")
        print("---")
        print("Wynik detekcji sygnału:")
        print("-> [False True False]  # Środkowa kolumna zawiera 1")

    def detekcja_brakow_isnan(self):
        print("INFO: DETEKCJA WARTOŚCI PUSTYCH (NaN)")
        print("-> A = np.array([[3, 2, 1, np.nan], [5, np.nan, 1, 6]])")
        print("---")
        print("Identyfikacja braków danych:")
        print("-> maska_nan = np.isnan(A)    # Zwraca maskę boolean")
        print("-> liczba_nan = np.sum(maska_nan)")
        print("---")
        print("Maska wyników:")
        print("-> [[False False False  True]\n    [False  True False False]]")
        print("UWAGA: NaN nie jest równy samemu sobie (np.nan == np.nan zwraca False).")

    def porownanie_bliskosci_allclose(self):
        print("INFO: PORÓWNYWANIE FLOATING-POINT")
        print("-> A = np.array([0.4, 0.5, 0.3])")
        print("-> B = np.array([0.39999999, 0.5000001, 0.3])")
        print("---")
        print("Bezpośrednie porównanie (==) jest zawodne:")
        print("-> A == B  # Zwróci False przez błędy precyzji")
        print("---")
        print("Bezpieczne porównanie z tolerancją:")
        print("-> np.allclose(A, B)  # Sprawdza czy wartości są 'bliskie'")
        print("---")
        print("Wynik (np.allclose): True")

    def operacje_logiczne_greater(self):
        print("INFO: OPERACJE LOGICZNE I MASKOWANIE")
        print("-> A = np.array([0.4, 0.5, 0.3, 0.9])")
        print("-> B = np.array([0.38, 0.51, 0.3, 0.91])")
        print("---")
        print("Wektorowe operacje porównania:")
        print("-> maska = A > B              # Operator standardowy")
        print("-> np.greater(A, B)           # Funkcjonalny odpowiednik")
        print("---")
        print("Wynik maskowania:")
        print("-> [ True False False False]")


class GeneratorStaly(__BazaNauki__, GeneratorStaly_Protocol):
    opis_menu = "inicjalizacja_zeros_ones_full, sekwencje_arange_reshape_eye, podzial_linspace"
    
    """
    inicjalizacja_zeros_ones_full - tworzenie macierzy zer, jedynek i stałych
    sekwencje_arange_reshape_eye - generowanie sekwencji i macierzy jednostkowej
    podzial_linspace - tworzenie równomiernych przedziałów liczbowych
    """

    def inicjalizacja_zeros_ones_full(self):
        print("INFO: TWORZENIE MACIERZY INICJALIZACYJNYCH")
        print("-> zeros = np.zeros((4, 4), dtype=int)  # Macierz zer")
        print("---")

        print("Wypełnianie stałą wartością (np. dla obrazów):")
        print("-> full_255 = np.full((3, 3), 255, int)")
        print("-> ones_skalar = np.ones((3, 3), int) * 255")
        print("---")
        print("Wynik operacji np.full:")
        print(f"-> {np.full((2, 2), 255)}")
        print("UWAGA: Ustawienie poprawnego <dtype> oszczędza pamięć RAM.")

    def sekwencje_arange_reshape_eye(self):
        print("INFO: SEKWENCJE I MACIERZ JEDNOSTKOWA")
        print("-> wektor = np.arange(10, 100, 1, int)  # Od 10 do 99")
        print("-> macierz = wektor.reshape(9, 10)      # Zmiana kształtu")
        print("---")

        print("Macierz jednostkowa (Algebra liniowa):")
        print("-> jednostkowa = np.eye(6, 6)           # 1 na przekątnej")
        print("---")
        print(
            "INFO: np.eye jest kluczowa przy inicjalizacji wag w sieciach neuronowych."
        )

    def podzial_linspace(self):
        print("INFO: TWORZENIE RÓWNOMIERNYCH PRZEDZIAŁÓW")
        print("---")
        print("Generowanie punktów (np. do wykresów):")
        print("-> linia = np.linspace(0, 1, 11)  # 0 do 1, dokładnie 11 kroków")
        print("---")
        print("Wynik np.linspace:")
        print(f"-> {np.linspace(0, 1, 5)}  # Przykład dla 5 kroków")
        print(
            "INFO: W przeciwieństwie do arange, linspace gwarantuje uwzględnienie punktu końcowego."
        )


class GeneratorLosowy(__BazaNauki__, GeneratorLosowy_Protocol):
    opis_menu = "rozklady_rand_normal, losowosc_default_rng_choice, struktury_diagonalne_diag"
    
    """
    rozklady_rand_normal - generowanie danych z rozkładów statystycznych
    losowosc_default_rng_choice - nowoczesna losowość i wybór próbek
    struktury_diagonalne_diag - szybkie tworzenie macierzy diagonalnych
    """

    def rozklady_rand_normal(self):
        print("INFO: PRÓBKOWANIE DANYCH LOSOWYCH")
        print("-> np.random.rand(5)  # Rozkład jednostajny [0, 1)")
        print("---")

        print("Rozkład normalny (Gaussa):")
        print("-> srednia, wariancja = 100, 5")
        print(
            "-> probki = np.random.normal(loc=<srednia>, scale=np.sqrt(<wariancja>), size=(5, 2))"
        )
        print("---")
        print("Wynik próbkowania:")
        print(f"-> {np.random.normal(100, np.sqrt(5), (2, 2))}")
        print("INFO: scale to odchylenie standardowe, nie wariancja.")

    def losowosc_default_rng_choice(self):
        print("INFO: NOWOCZESNA LOSOWOŚĆ (BitGenerators)")
        print("-> rng = np.random.default_rng(seed=42)  # Rekomendowane w 2026")
        print("-> pula = np.arange(1, 50)")
        print("---")

        print("Losowanie bez powtórzeń (np. Lotto):")
        print("-> numbers = rng.choice(pula, 5, replace=False)")
        print("---")
        print(
            "UWAGA: Używanie obiektu <rng> zapewnia powtarzalność badań statystycznych."
        )

    def struktury_diagonalne_diag(self):
        print("INFO: STRUKTURY DIAGONALNE")
        print("-> x = np.arange(6)     # Wektor bazowy")
        print("-> diagonalna = np.diag(x)")
        print("---")
        print("Macierz wynikowa:")
        print(f"-> {np.diag(np.arange(3))}  # Fragment macierzy")
        print("---")
        print(
            "INFO: np.diag() może też służyć do wyciągania przekątnej z istniejącej macierzy."
        )


class MagazynDanych(__BazaNauki__, MagazynDanych_Protocol):
    opis_menu = "trwalosc_save_load_savetxt"
    
    """
    trwalosc_save_load_savetxt - serializacja i eksport danych do plików
    """

    def trwalosc_save_load_savetxt(self):
        print("INFO: SERIALIZACJA I EKSPORT DANYCH")
        print("-> x = np.arange(12).reshape(3, 4)")
        print("---")

        print("Format binarny (.npy) - Najszybszy:")
        print("-> np.save('array.npy', x)        # Zapisuje kształt i typ")
        print("-> plik_npy = np.load('array.npy')")
        print("---")

        print("Format tekstowy (.txt) - Czytelny dla Excela:")
        print("-> np.savetxt('array.txt', X=x, fmt='%0.2f')")
        print("-> y = np.loadtxt('array.txt').astype(int)")
        print("---")

        print("Konwersja do struktur Pythona:")
        print("-> lista_python = x.tolist()      # Zwraca standardowe list()")
        print("---")
        print(
            "UWAGA: Format .npy jest specyficzny dla NumPy i nie otworzysz go w Notatniku."
        )
        print("INFO: savetxt pozwala na precyzyjne ustawienie formatu danych <fmt>.")


class MacierzowaGeometria(__BazaNauki__, MacierzowaGeometria_Protocol):
    opis_menu = "otaczanie_marginesem_pad, zmiana_ukladu_reshape_view, okna_przesuwne_sliding_window_view, ekstrakcja_sliding_window_view, zarzadzanie_pamiecia_copy"
    
    """
    otaczanie_marginesem_pad - tworzenie marginesów (padding) dla macierzy
    zmiana_ukladu_reshape_view - edycja sub-regionów przez widoki 4D
    okna_przesuwne_sliding_window_view - generowanie przesuwnych okien lokalnych
    ekstrakcja_sliding_window_view - analiza i mapowanie współrzędnych okien
    zarzadzanie_pamiecia_copy - różnice między widokiem a kopią edytowalną
    """

    def otaczanie_marginesem_pad(self):
        print("INFO: ARCHITEKTURA MACIERZY (np.pad)")
        print("-> matrix = np.ones((4, 4))")
        print("---")
        print("Techniki paddingu:")
        print("-> A = np.pad(matrix, 1)                      # Ramka zer")
        print("-> B = np.pad(matrix, 2, constant_values=3)    # Gruba ramka 3x3")
        print("---")
        print("Padding asymetryczny <height_pad>, <width_pad>:")
        print("-> C = np.pad(matrix, ((1, 2), (3, 4)))")
        print("UWAGA: Pozwala dopasować wymiary macierzy do rozmiaru kernela.")

    def zmiana_ukladu_reshape_view(self):
        print("INFO: WIDOKI BLOKOWE (RESHAPE AS VIEW)")
        print("-> A = np.zeros((6, 6))")
        print("-> view = A.reshape(3, 2, 3, 2)  # Rzutowanie 2D na 4D")
        print("---")
        print("Modyfikacja sektorowa bez pętli:")
        print("-> view[:, 0, :, 0] = 10  # Edycja konkretnych pod-siatek")
        print("-> view[:, 1, :, 0] = 5")
        print("---")
        print("Siatka 10x10 (Sektorowa modyfikacja):")
        print("-> v2 = B.reshape(2, 5, 2, 5)")
        print("-> v2[:, 0, 0, 0] = 7     # Punkt startowy każdego sektora")

    def okna_przesuwne_sliding_window_view(self):
        print("INFO: MECHANIKA OKNA PRZESUWNEGO")
        print("-> A = np.arange(16).reshape(4, 4)")
        print("-> windows = sliding_window_view(A, (2, 2))")
        print("---")
        print("Analiza okien:")
        print("-> Kształt widoku: (3, 3, 2, 2)  # 3x3 okna o rozmiarze 2x2")
        print("-> windows[0, 0]    # Lewy górny róg")
        print("-> windows[-1, -1]  # Prawy dolny róg")

    def ekstrakcja_sliding_window_view(self):
        print("INFO: STRUKTURA MACIERZY 4D (WINDOW MAPPING)")
        print("-> windows = sliding_window_view(A, (3, 3))")
        print("---")
        print("System współrzędnych (k, l, i, j):")
        print("-> k, l # Pozycja okna na macierzy źródłowej")
        print("-> i, j # Indeksy wewnątrz lokalnego okna")
        print("---")
        print("Ekstrakcja cech:")
        print("-> windows[0, 0]       # Pierwsze sąsiedztwo (Patch)")
        print("-> windows[:, :, 1, 1] # Wszystkie punkty centralne okien")

    def zarzadzanie_pamiecia_copy(self):
        print("INFO: MODYFIKACJA BUFORA PAMIĘCI")
        print("-> windows = sliding_window_view(A, (3, 3))")
        print("---")
        print("Zarządzanie flagami zapisu:")
        print("-> editable = windows.copy()  # Tworzenie niezależnej kopii")
        print("-> editable[:, :, 1, 1] = 99  # Bezpieczna edycja")
        print("---")
        print("UWAGA: Oryginalny <A> pozostaje nienaruszony przy edycji kopii.")
        print("INFO: Widoki (Views) standardowo mają flagę WRITEABLE=False.")


class ProcesorAlgorytmow(__BazaNauki__, ProcesorAlgorytmow_Protocol):
    opis_menu = "mnozenie_macierzowe_einsum, pooling_window_plus_einsum"
    
    """
    mnozenie_macierzowe_einsum - implementacja splotu notacją Einsteina
    pooling_window_plus_einsum - agregacja danych i pooling przestrzenny
    """

    def mnozenie_macierzowe_einsum(self):
        print("INFO: Splot macierzowy (EINSUM CONVOLUTION)")
        print("-> A = np.arange(25).reshape(5, 5)")
        print("-> windows = sliding_window_view(A, (3, 3))")

        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        print(f"-> Kernel (Filtr Sobel Vertical): \n{kernel}")
        print("---")

        # Operacja splotu: ij (kernel) * klij (widok okien) -> kl (mapa cech)
        result = np.einsum(
            "ij,klij->kl", kernel, np.arange(25).reshape(5, 5).view()
        )  # Uproszczone dla prezentacji

        print("Wykonano operację na strukturze okien:")
        print("-> result = np.einsum('ij,klij->kl', <kernel>, <windows>)")
        print(f"Mapa cech (Feature Map):\n{result}")
        print("---")
        print("INFO: ij redukuje wymiary okna przez sumowanie iloczynów skalarnych.")

    def pooling_window_plus_einsum(self):
        print("INFO: POOLING I AGREGACJA PRZESTRZENNA")
        print("-> A = np.arange(25).reshape(5, 5)")
        print("---")

        print("Redukcja średnią (Average Pooling):")
        print("-> blur = np.einsum('klij->kl', <windows>) / 9  # Sumowanie okien")

        print("\nMax Pooling ze skokiem (Stride=2):")
        print("-> stride_windows = windows[::2, ::2]         # Slicing co 2")
        print("-> max_pool = np.max(stride_windows, axis=(2, 3))")
        print("---")
        print(
            "UWAGA: Pooling służy do ekstrakcji najsilniejszych cech i redukcji wymiarów."
        )


class AnalizaIZbiory(__BazaNauki__, AnalizaIZbiory_Protocol):
    opis_menu = "detekcja_maksimow_argmax, unikalnosc_unique_all, czesci_wspolne_intersect1d, laczenie_macierzy_append"
    
    """
    detekcja_maksimow_argmax - wyszukiwanie indeksów maksimów
    unikalnosc_unique_all - znalezienie unikalnych kolumn i indeksów
    czesci_wspolne_intersect1d - znajdowanie części wspólnej zbiorów
    laczenie_macierzy_append - dodawanie nowych wierszy/kolumn do macierzy
    """

    def detekcja_maksimow_argmax(self):
        print("Analiza ekstremów w macierzy (np.argmax):")
        print("-> A = np.array([[1, 8, 5], [3, 2, 9]])")
        print(f"-> Globalny max (indeks): 5")
        print(f"-> Max w kolumnach: [1 0 1]  # axis=0")
        print(f"-> Max w wierszach: [1 2]    # axis=1")
        print("---")
        print("INFO: argmax zwraca indeks spłaszczony, jeśli nie określono osi.")

    def unikalnosc_unique_all(self):
        print("Weryfikacja unikalności danych (np.unique):")
        print("-> A = np.array([[1, 0, 1], [5, 2, 5]])")
        print(
            "-> u_cols, idx = np.unique(A, axis=1, return_index=True)  # Unikalne kolumny"
        )
        print("---")
        print("Wynik czyszczenia danych:")
        print("-> Unikalne kolumny:\n[[0 1]\n [2 5]]")
        print(f"-> Indeksy: [1 0]  # Pozycje pierwszych wystąpień")

    def czesci_wspolne_intersect1d(self):
        print("Operacje na zbiorach (np.intersect1d):")
        print("-> A = np.arange(8)           # [0 1 2 3 4 5 6 7]")
        print("-> B = np.array([6, 7, 8, 9])")
        print("---")
        print(f"-> Część wspólna <A> i <B>: [6 7]")
        print("UWAGA: Funkcja zawsze zwraca posortowaną tablicę wyników.")

    def laczenie_macierzy_append(self):
        print("Łączenie struktur danych (np.append):")
        print("-> A = np.ones((2, 2))")
        print("-> B = np.zeros((1, 2))")
        print("-> C = np.append(A, B, axis=0)")
        print("---")
        print("Macierz po połączeniu wzdłuż wierszy:")
        print("-> [[1. 1.]\n    [1. 1.]\n    [0. 0.]]")


if __name__ == "__main__":
    moje_lekcje = [
        WalidacjaIDetekcja,
        GeneratorStaly,
        GeneratorLosowy,
        MagazynDanych,
        MacierzowaGeometria,
        ProcesorAlgorytmow,
        AnalizaIZbiory,
    ]

    __BazaNauki__(interaktywne=True, lista_klas=moje_lekcje)
