## Skrypt inicjalizacyjny modułu __init__.py
* **Główna odpowiedzialność**: Dynamiczne filtrowanie oraz eksponowanie obiektów zaimportowanych z modułu wewnętrznego .baza_nauki poprzez jawną definicję zmiennej globalnej __all__.
* **Stan obiektu (Atrybuty self):**
* **Brak:** Skrypt nie definiuje instancji klasy ani atrybutów stanu self.
* **Szczegółowy opis metod:**
* **Brak:** Skrypt wykonuje operacje proceduralne w przestrzeni nazw modułu i nie posiada zdefiniowanych metod.
* **Zależności i ograniczenia:** Wymaga istnienia relatywnego modułu .baza_nauki. Wykorzystuje standardową bibliotekę types. Po zakończeniu wykonania usuwa zmienne tymczasowe base_attrs oraz imported_names przy użyciu instrukcji del, czyszcząc przestrzeń lokalną modułu. Moduł eksponuje tylko te nazwy, które zawierają znak podkreślenia _ lub zaczynają się od __, a nie należą do domyślnych atrybutów obiektów typu types.ModuleType.
* **Kontrakt danych (Wejście/Wyjście):** Wejście: Przestrzeń nazw zaimportowana instrukcją from .baza_nauki import *. Wyjście: Zmienna __all__ typu list[str] definiująca publiczny interfejs modułu.


# KLASA: BazaNauki
* **Główna odpowiedzialność:** Dynamiczne zarządzanie sekwencją wykonywania publicznych metod biznesowych na podstawie introspekcji, dokumentacji tekstowej (__doc__) lub filtrów, a także realizacja logiki interfejsu konsolowego w trybie interaktywnym bądź standardowym.
* **Stan obiektu (Atrybuty self):**
* **edukator:** type - Referencja do zewnętrznej klasy lub obiektu globalnego Edukator zarządzającego sesją interaktywną.
* **lista_klas:** Optional[list[type]] - Zbiór obiektów klas przekazywany jako kontekst wykonawczy do silnika interaktywnego.
* **wybrane_metody:** list[str] - Uporządkowana sekwencja nazw metod przeznaczonych do automatycznego, sekwencyjnego wywołania.
* **Szczegółowy opis metod:**
* **__init__(aktywne, pokaz_docstring, czekaj_na_enter, czysc_ekran, interaktywne, tylko_dane, lista_klas, wybrane_metody_f) -> None:** Inicjalizuje instancję, parsuje __doc__ w celu ustalenia kolejności metod, aplikuje filtry wejściowe i automatycznie uruchamia jeden z dwóch wewnętrznych silników prezentacyjnych.
* **_silnik_interaktywny() -> None:** Uruchamia sesję interaktywną poprzez wywołanie metody start na obiekcie edukatora, przekazując jawną listę klas lub typ instancji własnej.
* **_rysuj_opis(aktywny_idx: Optional[int]) -> None:** Parsuje linie dokumentacji __doc__, renderuje ramkę tekstową ASCII w strumieniu wyjściowym konsoli oraz wyróżnia opcjonalny aktywny krok za pomocą kodów ANSI ESCAPE (\033[1;32m).
* **_silnik_standardowy(pokaz_docstring: bool, czysc_ekran: bool, czekaj_na_enter: bool) -> None:** Wykonuje pętlę sekwencyjnego wywoływania metod za pomocą refleksji getattr, opcjonalnie czyszcząc bufor ekranu i wstrzymując wątek instrukcją input.
* **Zależności i ograniczenia:** Wymaga zdefiniowanego w wyższym zasięgu modułu obiektu Edukator oraz biblioteki os do czyszczenia ekranu (cls / clear). Wykorzystuje kody ANSI do formatowania tekstu terminala. Ukrytym efektem ubocznym inicjalizacji jest natychmiastowe przejęcie przepływu sterowania, uruchomienie pętli przetwarzania i modyfikacja strumienia wyjściowego konsoli.
* **Kontrakt danych (Wejście/Wyjście):** Wejście: Parametry konfiguracyjne typu bool, opcjonalne kolekcje list[type], list[str] oraz indeks numeryczny int. Wyjście: Instancja klasy z zainicjalizowanym stanem lub brak (efekty uboczne w postaci interakcji I/O ze strumieniami sys.stdout/sys.stdin).


## KLASA: Edukator
* **Główna odpowiedzialność**: Realizacja wielopoziomowego interfejsu konsolowego w czasie rzeczywistym z obsługą nawigacji klawiszowej, segmentacją opisów struktur danych oraz dynamicznym nakładaniem kolorowania składni tekstu.
* **Stan obiektu (Atrybuty self):**
* **Brak:** Klasa nie implementuje metody __init__ ani nie przechowuje stanu instancji (zawiera wyłącznie metody statyczne i klasowe).
* **Szczegółowy opis metod:**
* **nawiguj(tytul: str, opcje: list, doc_rysuj_func: Optional[Callable], start_idx: int) -> int:** Implementuje pętlę zdarzeń przechwytującą surowe bajty klawiatury, przemieszcza kursor terminala w pionie za pomocą sekwencji VT100 (\033{wysokosc_powrotu}A), nadpisuje zawartość ekranu i zwraca indeks wybranej opcji lub -1.
* **start(lista_klas: list) -> None:** Analizuje atrybuty opis_menu klas, dzieli tekst na segmenty poniżej 50 znaków, inicjalizuje instancje z flagą tylko_dane=True, a następnie podmienia globalną funkcję wbudowaną builtins.print na autorską metodę cls.koloruj_tekst w bloku try-finally.
* **koloruj_tekst(*args, **kwargs) -> None:** Przekształca argumenty pozycyjne na ciąg znaków, a następnie stosuje kaskadowe dopasowania regex (re.sub) w celu wyróżnienia słów kluczowych, komentarzy, literałów czy błędów, po czym wysyła przetworzony strumień do sys.stdout.
* **Zależności i ograniczenia:** Ściśle uzależniona od biblioteki msvcrt (funkcja getch), co ogranicza działanie do systemów Windows. Wykorzystuje moduły os, sys, re oraz builtins do manipulacji globalnym zachowaniem funkcji print. Wywołuje silne efekty uboczne: ukrywanie kursora systemowego (\033?25l), czyszczenie bufora ekranu (cls/clear), blokowanie wątku na wejściu I/O oraz tymczasową modyfikację środowiska uruchomieniowego Pythona.
* **Kontrakt danych (Wejście/Wyjście):** Wejście: Lista obiektów klas list[type], łańcuchy tekstowe (str), listy opcji list[str], opcjonalne wywołania Callable, indeksy int oraz argumenty zmiennotypowe *args i **kwargs. Wyjście: Liczba całkowita int (indeks menu lub -1 jako sygnał wyjścia) lub brak wartości zwracanej (efekty uboczne w strumieniach I/O terminala).


## Komponenty Dekoratorów Modułu
* **Główna odpowiedzialność:** Automatyzacja zarządzania środowiskiem plików tymczasowych oraz masowe, dynamiczne nakładanie zachowań cross-cutting (aspektów) na publiczne metody klas za pomocą mechanizmów refleksji.
* **Stan obiektu (Atrybuty self):**
* **Brak:** Funkcje działają w paradygmacie funkcyjnym i nie definiują instancji klasy ani atrybutów stanu self.
* **Szczegółowy opis metod:**
* **bezpieczny_wrapper(func: Callable) -> Callable:** Tworzy domknięcie realizujące operacje idempotencji środowiska dyskowego poprzez sprawdzenie obecności, usunięcie (shutil.rmtree) i ponowne utworzenie katalogu tmp bezpośrednio przed wywołaniem owiniętej funkcji.
* **dekoruj_wszystko(*dekoratory: Callable) -> Callable:** Fabryka dekoratorów klas, która iteruje po słowniku atrybutów struktury (vars(cls)), filtruje metody publiczne i aplikuje na nie sekwencyjnie stos przekazanych dekoratorów za pomocą setattr.
* **Zależności i ograniczenia:** Wymaga modułów standardowych os, shutil oraz functools.wraps. Wywołuje silne, ukryte efekty uboczne w systemie plików (I/O) poprzez destruktywne usuwanie i tworzenie zasobów w relatywnej ścieżce dyskowej ./tmp przy każdym uruchomieniu owiniętej metody. Ograniczeniem jest mutowanie struktury klas w miejscu (in-place) podczas ładowania modułu.
* **Kontrakt danych (Wejście/Wyjście):** Wejście: Obiekty wywoływalne Callable (funkcje, dekoratory) oraz referencja typu klasowego type. Wyjście: Zmodyfikowany obiekt klasy type lub opakowana funkcja Callable zachowująca oryginalne metadane sygnatury.
