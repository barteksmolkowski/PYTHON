## KLASA: FileManager
* Główna odpowiedzialność: Zarządzanie operacjami na strukturze systemu plików, w tym listowaniem zawartości, tworzeniem katalogów pojedynczych i zagnieżdżonych, zmianą nazw oraz ekstrakcją metadanych ścieżek dostępu.
* Stan obiektu (Atrybuty self):
* Brak jawnych atrybutów: Wszystkie stany oraz parametry konfiguracyjne są dziedziczone z klasy bazowej __BazaNauki__.
* Szczegółowy opis metod:
* **list_files_and_dirs() -> None**: Iteruje po zawartości bieżącego katalogu roboczego pobranej przez os.listdir, a następnie klasyfikuje i wypisuje każdy element jako plik (os.path.isfile) lub folder (os.path.isdir`).
* **create_directory() -> None**: Sprawdza istnienie docelowego katalogu o nazwie 'pliki' za pomocą os.path.exists, po czym dokonuje jego alokacji przez os.mkdir` wyłącznie przy braku wcześniejszego istnienia struktury.
* **rename_item() -> None**: Prezentuje instrukcję użycia wbudowanej funkcji systemowej os.rename` przeznaczonej do modyfikacji nazw obiektów w drzewie katalogów.
* **path_info() -> None**: Parsuje predefiniowany ciąg tekstowy ścieżki w celu ekstrakcji i wyświetlenia katalogu nadrzędnego (os.path.dirname), nazwy pliku (os.path.basename) oraz pełnej ścieżki bezwzględnej (os.path.abspath`).
* **create_nested_dirs_and_file() -> None**: Wyznacza strukturę katalogów nadrzędnych, tworzy je rekurencyjnie z użyciem os.makedirs(z flagą tolerancji istnieniaexist_ok=True), a następnie inicjalizuje pusty plik tekstowy za pomocą wbudowanego menedżera kontekstu with open`.
* Zależności i ograniczenia: Dziedziczy bezpośrednio po klasie __BazaNauki__ z pakietu a01_Nauka_Aktualna. Wszystkie metody są modyfikowane przez dekorator @bezpieczny_wrapper obsługujący wyjątki operacji dyskowych. Klasa ściśle zależy od wbudowanego modułu systemowego os. Operacje wejścia/wyjścia (I/O) generują trwałe efekty uboczne w strukturze plików systemu operacyjnego hosta.
* Kontrakt danych (Wejście/Wyjście): Wejście: Brak (metody operują na stałych lokalnych i bieżącym środowisku OS), Wyjście: None (wyniki działań oraz logi operacji są przekazywane na standardowe wyjście konsoli).


## KLASA: brak / Kod Proceduralny (Funkcja Główna: przeszukiwanie)
* Główna odpowiedzialność: Realizacja wielokryteriowego algorytmu przeszukiwania struktury drzewa katalogów systemu operacyjnego w celu identyfikacji plików na podstawie fraz kluczowych i rozszerzeń, z zintegrowanym systemem buforowania dyskowego oraz profilowania wydajnościowego.
* Stan obiektu (Atrybuty self):
* Brak klasy: Analizowany kod zawiera wyłącznie wolne funkcje oraz dekoratory i nie definiuje instancji obiektowych (brak atrybutów self).
* Szczegółowy opis metod:
* **czasFunkcji(func: Callable) -> Callable**: Dekorator profilujący, który mierzy czas wykonywania funkcji opakowanej za pomocą time()`, a następnie zwraca rozszerzoną krotkę zawierającą listę dopasowań, licznik operacji oraz obliczoną deltę czasową.
* **zapisPlikow(func: Callable) -> Callable**: Dekorator persystencji realizujący asynchroniczny zapis nowo zidentyfikowanych ścieżek do pliku tekstowego pamiecPlikow.txt w kodowaniu UTF-8 przy użyciu trybu dopisywania (a`).
* **przygotowanie(func: Callable) -> Callable**: Dekorator konfiguracyjny modyfikujący parametr maxWynikow` poprzez inkrementację jego wartości o 1 przed przekazaniem sterowania do funkcji właściwej.
* **odbiorZapisu(func: Callable) -> Callable**: Dekorator wejściowy sprawdzający istnienie pliku pamięci podręcznej za pomocą os.path.exists, parsujący jego zawartość metodą readlines()do listy struktur tekstowych i wstrzykujący ją jako parametrpamiec`.
* **przeszukiwanie(lista_fraz: Union[list, str], odKonca: str, zaczynaOd: str, maxWynikow: int, pamiec: list) -> Tuple**: Przeszukuje rekurencyjnie system plików od punktu startowego przy użyciu generatora os.walk. Normalizuje wejściowe frazy do małych liter (lower()), filtruje pliki za pomocą algorytmu dopasowania kwantyfikatora all(), weryfikuje sufiks (endswith), odrzuca duplikaty znajdujące się w strukturze pamieci przerywa pętlę po osiągnięciu progumaxWynikow`.
* Zależności i ograniczenia: Ścisła zależność od modułów standardowych os, functools.wraps oraz time.time. Operacje wejścia/wyjścia (I/O) generują efekty uboczne w postaci odczytu i zapisu pliku pamiecPlikow.txt. Algorytm jest synchroniczny i blokujący, a jego wydajność na dyskach HDD/SSD zależy od uprawnień dostępu i głębokości drzewa katalogów (potencjalne błędy PermissionError przy braku uprawnień administratora systemu).
* Kontrakt danych (Wejście/Wyjście): Wejście: lista_fraz: list/str, odKonca: str, zaczynaOd: str, maxWynikow: int, pamiec: list/None. Wyjście: krotka (Tuple) zawierająca: [katalogi: list, sprawdzone: int, czas_wykonania: float].
