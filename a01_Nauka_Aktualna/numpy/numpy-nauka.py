from typing import Protocol

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from PYTHON_NAUKA import __BazaNauki__


class __WalidacjaIDetekcja__(Protocol):
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
class __FabrykaDanych__(Protocol):
    """
    SEKCJA GENEROWANIA:
    inicjalizacja_zeros_ones_full         -> np.zeros, np.ones, np.full
    sekwencje_arange_reshape_eye          -> np.arange, np.reshape, np.eye
    rozklady_rand_normal                  -> np.random.rand, np.random.normal
    podzial_linspace                      -> np.linspace
    losowosc_default_rng_choice           -> np.random.default_rng, rng.choice
    struktury_diagonalne_diag             -> np.diag
    """
    def inicjalizacja_zeros_ones_full(self) -> None: ...
    def sekwencje_arange_reshape_eye(self) -> None: ...
    def rozklady_rand_normal(self) -> None: ...
    def podzial_linspace(self) -> None: ...
    def losowosc_default_rng_choice(self) -> None: ...
    def struktury_diagonalne_diag(self) -> None: ...
class __MagazynDanych__(Protocol):
    """
    SEKCJA PLIKÓW:
    trwalosc_save_load_savetxt            -> np.save, np.load, np.savetxt, np.loadtxt
    """
    def trwalosc_save_load_savetxt(self) -> None: ...
class __MacierzowaGeometria__(Protocol):
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
    def zarzadzanie_pamiecia_copy(self) -> None: ...
class __ProcesorAlgorytmow__(Protocol):
    """
    SEKCJA ALGORYTMÓW:
    ekstrakcja_sliding_window_view        -> np.lib.stride_tricks.sliding_window_view
    mnozenie_macierzowe_einsum            -> np.einsum
    pooling_window_plus_einsum            -> sliding_window_view + einsum
    """
    def ekstrakcja_sliding_window_view(self) -> None: ...
    def mnozenie_macierzowe_einsum(self) -> None: ...
    def pooling_window_plus_einsum(self) -> None: ...

class WalidacjaIDetekcja(__WalidacjaIDetekcja__, __BazaNauki__):
    def weryfikacja_logiczna_all(self):
        """
        Sprawdzanie, czy WSZYSTKIE elementy wzdłuż osi spełniają warunek (Logiczne AND).
        W 2026 kluczowe przy walidacji batchy danych przed wejściem do modelu.
        """
        print(f"\n=> WERYFIKACJA LOGICZNA: np.all <=")

        A = np.array([[3, 2, 1, 4], [5, 2, 1, 6]])
        B = np.array([[3, 2, 1, 4], [5, 2, 0, 6]])
        C = np.array([[True, False, False], [True, True, True]])

        lista = [A, B, C]
        nazwy = ["A (same >0)", "B (zawiera 0)", "C (Boolean)"]

        print("Czy wszystkie elementy w rzędach (axis=1) są True / niezerowe?")
        for nazwa, matrix in zip(nazwy, lista):
            # axis=1 sprawdza każdy wiersz osobno
            wynik = np.all(matrix, axis=1)
            print(f"{nazwa}: {wynik}")
    def weryfikacja_logiczna_any(self):
        """
        Sprawdzanie, czy JAKIKOLWIEK element wzdłuż osi spełnia warunek (Logiczne OR).
        Często używane do wykrywania anomalii lub sygnałów w kanałach.
        """
        print(f"\n=> WERYFIKACJA LOGICZNA: np.any <=")

        A = np.array([[0, 0, 0], [0, 0, 0]])
        B = np.array([[0, 0, 0], [0, 1, 0]])
        C = np.array([[False, False, False], [True, False, False]])
        D = np.array([[0.1, 0.0]])

        lista = [A, B, C, D]
        nazwy = ["A (same 0)", "B (ma 1)", "C (ma True)", "D (float)"]

        print("Czy w kolumnach (axis=0) występuje choć jedna wartość True / != 0?")
        for nazwa, matrix in zip(nazwy, lista):
            # axis=0 sprawdza każdą kolumnę redukując wiersze
            wynik = np.any(matrix, axis=0)
            print(f"{nazwa}: {wynik}")
    def detekcja_brakow_isnan(self):
        """
        Identyfikacja wartości nieokreślonych (Not a Number).
        Podstawa preprocessingu danych w 2026 roku.
        """
        print(f"\n=> DETEKCJA WARTOŚCI PUSTYCH (NaN) <=")

        A = np.array([[3, 2, 1, np.nan],
                    [5, np.nan, 1, 6]])

        # np.isnan zwraca maskę boolean o tym samym kształcie co oryginał
        maska_nan = np.isnan(A)
        
        print(f"Macierz wejściowa:\n{A}")
        print(f"Maska wartości pustych (np.isnan):\n{maska_nan}")
        print(f"Liczba NaN w macierzy: {np.sum(maska_nan)}")
    def porownanie_bliskosci_allclose(self):
        """
        Bezpieczne porównywanie liczb zmiennoprzecinkowych z tolerancją.
        """
        print(f"\n=> PORÓWNYWANIE FLOATING-POINT <=")

        A = np.array([0.4, 0.5, 0.3])
        B = np.array([0.39999999, 0.5000001, 0.3])
        
        # Bezpośrednie porównanie (==) jest zawodne przez błędy precyzji
        print(f"Porównanie indeksowe (==): {A == B}") 
        
        # np.allclose to standard w 2026: sprawdza, czy wartości są "bliskie" (domyślna tolerancja)
        print(f"Porównanie tolerancyjne (np.allclose): {np.allclose(A, B)}")
    def operacje_logiczne_greater(self):
        """
        Wektorowe operacje logiczne (element-wise comparisons) zwracające maski boolean.
        """
        print(f"\n=> OPERACJE LOGICZNE I MASKOWANIE <=")

        A = np.array([0.4, 0.5, 0.3, 0.9])
        B = np.array([0.38, 0.51, 0.3, 0.91])

        # Standardowe operatory: zwracają maskę boolean
        maska_wiekszosci = (A > B)
        print(f"Czy idx A > idx B?: {maska_wiekszosci}")
        
        # Funkcjonalny odpowiednik (ufunc)
        print(f"np.greater(A, B): {np.greater(A, B)}")
class FabrykaDanych(__FabrykaDanych__, __BazaNauki__):
    def inicjalizacja_zeros_ones_full(self):
        """
        Szybkie inicjowanie macierzy zerami, jedynkami lub stałymi wartościami.
        """
        print(f"\n=> TWORZENIE MACIERZY INICJALIZACYJNYCH <=")

        # Macierz zer (typ int)
        zeros = np.zeros((4, 4), dtype=int)
        print(f"np.zeros((4,4)):\n{zeros}")

        # Macierz wypełniona stałą wartością (np. 255 dla obrazów)
        full_255 = np.full((3, 3), 255, int)
        print(f"np.full((3, 3), 255):\n{full_255}")

        # Alternatywa dla full: ones * skalar
        matrix_of_ones = np.ones((3, 3), int)
        print(f"np.ones() * 255:\n{matrix_of_ones * 255}")
    def sekwencje_arange_reshape_eye(self):
        """
        Tworzenie sekwencji numerycznych (arange) i macierzy jednostkowej (eye).
        """
        print(f"\n=> SEKWENCJE I MACIERZ JEDNOSTKOWA <=")

        # Wektor od 10 do 99
        wektor = np.arange(10, 100, 1, int)
        macierz = wektor.reshape(9, 10)
        print(f"Macierz 9x10 (arange + reshape):\n{macierz}")

        # np.eye() - kluczowa dla algebry liniowej i inicjalizacji wag w NN
        print(f"Macierz jednostkowa 6x6 (np.eye):\n{np.eye(6, 6)}")
    def rozklady_rand_normal(self):
        """
        Generowanie danych z różnych rozkładów statystycznych.
        """
        print(f"\n=> PRÓBKOWANIE DANYCH LOSOWYCH <=")

        # Generowanie z rozkładu jednostajnego [0, 1)
        print(f"np.random.rand(5) [Rozkład jednostajny]:\n{np.random.rand(5)}")

        # Generowanie z rozkładu normalnego (Gaussa)
        # loc = średnia, scale = odchylenie standardowe (lub sqrt(wariancji))
        srednia, wariancja = 100, 5
        probki_normalne = np.random.normal(loc=srednia, scale=np.sqrt(wariancja), size=(5, 2))
        print(f"Próbki z rozkładu normalnego (Mean={srednia}, Var={wariancja}):\n{probki_normalne}")
    def podzial_linspace(self):
        """
        Efektywne przechodzenie przez struktury oraz nowoczesne 
        generowanie próbek losowych.
        """
        print(f"\n=> ITERACJA I GENEROWANIE DANYCH <=")

        # Tworzenie równomiernych przedziałów (np. do wykresów lub wag)
        linia = np.linspace(0, 1, 11)
        print(f"\n\nLinspace (0-1, 11 kroków):\n{linia}")
    def losowosc_default_rng_choice(self):
        """
        Zastosowanie Generatora (PCG64) zamiast starego interfejsu np.random.
        Standard zalecany w 2026 roku dla powtarzalności badań.
        """
        print(f"\n=> NOWOCZESNA LOSOWOŚĆ (BitGenerators) <=")

        # Nowy styl: Jawnym obiektem Generatora
        rng = np.random.default_rng(seed=42)
        pula = np.arange(1, 50)
        
        # Wybór bez powtórzeń (np. do symulacji Lotto)
        numbers = rng.choice(pula, 5, replace=False)
        print(f"Losowanie (RNG Object): {numbers}")
    def struktury_diagonalne_diag(self):
        """
        Szybkie tworzenie struktur diagonalnych.
        """
        print(f"\n=> STRUKTURY DIAGONALNE <=")
        x = np.arange(6)
        diagonalna = np.diag(x)
        print(f"Macierz diagonalna z wektora {x}:\n{diagonalna}")
class MagazynDanych(__MagazynDanych__, __BazaNauki__):
    def trwalosc_save_load_savetxt(self):
        """
        Zapis i odczyt danych: formaty binarne (.npy) oraz tekstowe (.txt).
        """
        print(f"\n=> SERIALIZACJA I EKSPORT DANYCH <=")

        x = np.arange(12).reshape(3, 4)

        # 1. Format binarny NumPy (Szybki, zachowuje typy i kształt)
        np.save("array.npy", x)
        plik_npy = np.load("array.npy")
        print(f"Odczytano z .npy:\n{plik_npy}")

        # 2. Format tekstowy (Czytelny dla człowieka/Excela)
        # fmt="%0.2f" wymusza zapis zmiennoprzecinkowy
        np.savetxt('array.txt', X=x, fmt="%0.2f")
        y = np.loadtxt('array.txt').astype(int)
        print(f"Odczytano z .txt (po rzutowaniu na int):\n{y}")

        # 3. Konwersja do natywnej listy Python
        lista_python = x.tolist()
        print(f"Konwersja do list(): {lista_python}")
class MacierzowaGeometria(__MacierzowaGeometria__, __BazaNauki__):
    def otaczanie_marginesem_pad(self):
        """
        Manipulacja strukturą macierzy: symetria, otaczanie (padding) 
        oraz rzutowanie wymiarów (N-D Views).
        """
        print(f"\n=>ARCHITEKTURA MACIERZY: PADDING I RZUTOWANIE <=")

        # 1. Padding - Tworzenie marginesów dla kerneli
        matrix = np.ones((4, 4), dtype=float)
        
        # Różne techniki paddingu
        A = np.pad(matrix, 1) # Standardowa ramka 0
        B = np.pad(matrix, 2, constant_values=3) # Gruba ramka z wartością 3
        
        # Padding asymetryczny: (góra, dół), (lewo, prawo)
        height_pad, width_pad = (1, 2), (3, 4)
        C = np.pad(matrix, (height_pad, width_pad))
        
        print(f"Padding asymetryczny (Kształt {C.shape}):\n{C}")
    def zmiana_ukladu_reshape_view(self):
        """
        Technika Reshape-to-View: Pozwala na edycję konkretnych 
        sub-regionów macierzy bez pętli for.
        """
        print(f"\n=>WIDOKI BLOKOWE (RESHAPE AS VIEW) <=")

        A = np.zeros((6, 6))
        # Rzutujemy 2D na 4D, aby uzyskać dostęp do konkretnych "pod-siatek"
        # (ilość_bloków_H, wielkość_bloku_H, ilość_bloków_W, wielkość_bloku_W)
        view = A.reshape(3, 2, 3, 2)
        
        # Modyfikacja co drugiego elementu w strukturze blokowej
        view[:, 0, :, 0] = 10
        view[:, 1, :, 0] = 5
        
        print(f"Macierz 6x6 po modyfikacji przez widok 4D:\n{A}")

        # Przykład skalowalny: Siatka 10x10
        h_m, h_l, w_m, w_l = (2, 5, 2, 5)
        B = np.zeros((10, 10))
        v2 = B.reshape(h_m, h_l, w_m, w_l)
        v2[:, 0, 0, 0] = 7  # Ustawia wartość na początku każdego głównego sektora
        print(f"\nSiatka 10x10 (Sektorowa modyfikacja):\n{B}")
    def okna_przesuwne_sliding_window_view(self):
        """
        Sliding window jako narzędzie do ekstrakcji cech lokalnych.
        """
        print(f"\n=>MECHANIKA OKNA PRZESUWNEGO (SLIDING WINDOW) <=")

        A = np.arange(16).reshape(4, 4)
        # Generujemy wszystkie możliwe okna 2x2 przesuwając się po macierzy 4x4
        windows = sliding_window_view(A, (2, 2))
        
        print(f"Oryginał 4x4:\n{A}")
        print(f"Kształt widoku okien: {windows.shape}") # Wynik: (3, 3, 2, 2)
        print(f"Pierwsze okno (lewy górny róg):\n{windows[0, 0]}")
        print(f"Ostatnie okno (prawy dolny róg):\n{windows[-1, -1]}")
    def zarzadzanie_pamiecia_copy(self):
        """
        Zarządzanie flagami zapisu i bezpieczna modyfikacja danych.
        Różnica między bezpośrednim widokiem (Read-Only) a kopią edytowalną.
        """
        print(f"\n\n=>MODYFIKACJA BUFORA PAMIĘCI (IN-PLACE VS COPY) <=")

        A = np.arange(25).reshape(5, 5)
        windows = sliding_window_view(A, (3, 3))

        # Standardowo sliding_window_view w 2026 zwraca widok z flagą WRITEABLE=False
        # Tworzymy kopię (deep copy) aby umożliwić niezależną edycję
        editable_windows = windows.copy()

        print("Akcja: Przypisanie stałej wartości do centrów wszystkich okien.")
        editable_windows[:, :, 1, 1] = 99

        print(f"Zmodyfikowany widok (Fragment):\n{editable_windows[0, 0]}")
        print(f"Oryginał (Nienaruszony):\n{A}")
class ProcesorAlgorytmow(__ProcesorAlgorytmow__, __BazaNauki__):        
    def ekstrakcja_sliding_window_view(self):
        """
        Analiza struktury 4D generowanej przez widok przesuwnego okna.
        Pozwala zrozumieć mapowanie współrzędnych obrazu na lokalne sąsiedztwa.
        """
        print(f"\n=>STRUKTURA MACIERZY 4D (WINDOW MAPPING) <=")
        
        A = np.arange(25).reshape(5, 5)
        print(f"Macierz wejściowa (Input Matrix):\n{A}")

        # Generowanie widoku
        windows = sliding_window_view(A, (3, 3))

        print(f"\nKształt widoku (Shape): {windows.shape}")
        print("TECHNICZNIE: System współrzędnych (k, l, i, j).")
        print("k, l -> Indeksy pozycjonowania okna na macierzy źródłowej.")
        print("i, j -> Indeksy wewnątrz lokalnego okna (piksele sąsiedztwa).")

        print("\nDOSTĘP DO PODMACIERZY:")
        print("windows[0, 0]    -> Pierwsze sąsiedztwo (Top-Left Patch).")
        print("windows[:, :, 1, 1] -> Wyodrębnienie wszystkich punktów centralnych.")
    def mnozenie_macierzowe_einsum(self):
        """
        Implementacja splotu (convolution) przy użyciu notacji Einsteina.
        Najszybsza metoda mnożenia filtrów przez wiele okien bez pętli.
        """
        print(f"\n\n=>SPLOT MACIERZOWY (EINSUM CONVOLUTION) <=")
        
        A = np.arange(25).reshape(5, 5)
        windows = sliding_window_view(A, (3, 3))
        
        # Filtr wykrywający pionowe zmiany (Sobel Vertical)
        kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        print(f"Kernel (Filtr): \n{kernel}")
        
        # Operacja splotu: ij (kernel) * klij (widok okien) -> kl (mapa cech)
        # i, j są redukowane przez sumowanie (iloczyn skalarny okna i filtra)
        result = np.einsum('ij,klij->kl', kernel, windows)

        print(f"Wykonano: np.einsum('ij,klij->kl', kernel, windows)")
        print(f"Mapa cech (Feature Map):\n{result}")
    def pooling_window_plus_einsum(self):
        """
        Agregacja danych przy użyciu skoku (stride) oraz funkcji statystycznych.
        Metoda stosowana do zmniejszania rozdzielczości i ekstrakcji najsilniejszych cech.
        """
        print(f"\n\n=>POOLING I AGREGACJA PRZESTRZENNA <=")
        
        A = np.arange(25).reshape(5, 5)
        windows = sliding_window_view(A, (3, 3))

        print("--- Redukcja średnią (Mean Blur / Average Pooling) ---")
        blur = np.einsum('klij->kl', windows) / 9
        print(f"Wynik uśredniania:\n{blur.astype(int)}")

        print("\n--- Max Pooling ze skokiem (Stride=2) ---")
        # Wybieramy okna nienakładające się poprzez slicing mapy k, l
        stride_windows = windows[::2, ::2]
        max_pool = np.max(stride_windows, axis=(2, 3))

        print(f"Wynik Max Pooling (zmniejszenie wymiaru):\n{max_pool}")

if __name__ == "__main__":
    WalidacjaIDetekcja(False)
    FabrykaDanych(False)
    MagazynDanych(False)
    MacierzowaGeometria(False)
    ProcesorAlgorytmow(False)

A = np.arange(12).reshape(-1, 4)
B = np.array([[4, 3, 7, 2],
              [0, 5, 2, 6]])

C = np.append(A, B, axis=0)
print(C)
###

A = np.arange(8).reshape(-1, 4)
B = np.array([[9, 10, 11, 3],
              [2, 8, 0, 9]])
C = np.intersect1d(A, B, return_indices=False)
print(C)
###
A = np.array([[1, 0, 1],
              [1, 2, 3],
              [1, 0, 1]])
zbior = np.unique_all(A)
print(f"unikalne: {zbior[0]}")
print(f"pierwsze wystapienia unikalnych: {zbior[1]}")
print(f"mapa indeksow unikalnych: {zbior[2]}")
print(f"liczba unikalnych: {zbior[3]}")
A = np.array([[1, 0, 1],
              [5, 2, 5]])

zbior = np.unique(
    A, 
    axis=1, 
    return_index=True, 
    return_inverse=True, 
    return_counts=True
)

unikalne, indeksy, odwrotnosc, liczniki = zbior
print(f"\n1. Macierz unikalnych kolumn (uniques):\n{zbior[0]}")
print(f"2. Indeksy pierwszych wystąpień (indices): {zbior[1]}")
print(f"3. Liczba powtórzeń każdej kolumny (inverse): {zbior[2]}")
print(f"4. Przepis na odbudowę (counts): {zbior[3]}")

A = np.array([[0.4, 0.3, 0.3],
              [0.1, 0.1, 0.8],
              [0.2, 0.5, 0.3]])
# print(f"np.argmax: {np.argmax(A, axis=1, )}")
print(f"\n=> POZYCJA MAKSIMUM: np.argmax <=")
A = np.array([[1, 8, 5],
                [3, 2, 9]])

print(f"Indeks największej liczby (globalnie): {np.argmax(A)}") # Wynik: 5 (liczba 9)
print(f"Indeksy max w kolumnach: {np.argmax(A, axis=0)}") # Wynik: [1, 0, 1]
print(f"Indeksy max w wierszach: {np.argmax(A, axis=1)}") # Wynik: [1, 2]

A = np.array([[0.4, 0.3, 0.3],
              [0.1, 0.1, 0.8],
              [0.2, 0.5, 0.3]])

print(np.arg)