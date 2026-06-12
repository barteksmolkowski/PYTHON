## Definicje protokołów systemowych Git
* Główna odpowiedzialność: Deklaracja statycznych interfejsów typowania (protokołów) wymuszających implementację zestawu metod operacyjnych dla podsystemów Git (podstawy, diff, log, undo, refactor, branch/merge, tag, cherry-pick) na potrzeby weryfikacji typów w czasie statycznej analizy kodu.
* Stan obiektu (Atrybuty self):
* Brak: Protokoły definiują wyłącznie sygnatury metod i nie przechowują stanu instancji ani atrybutów self.
* Szczegółowy opis metod:
* GitBasic_Protocol(Protocol): Definiuje kontrakt dla bazowych komend (wersjonowanie, konfiguracja, inicjalizacja, status, dodawanie, zatwierdzanie, synchronizacja).
* GitDiff_Protocol(Protocol): Definiuje kontrakt dla operacji porównywania zmian (statystyki, łatki, pamięć podręczna, rewizje, skróty relatywne).
* GitLog_Protocol(Protocol): Definiuje kontrakt dla przeglądania historii (filtrowanie, wyszukiwanie, wizualizacja grafów, zarządzanie aliasami).
* GitUndo_Protocol(Protocol): Definiuje kontrakt dla operacji cofania zmian (revert, przywracanie z hasha, tryby reset soft/mixed/hard).
* GitRefactor_Protocol(Protocol): Definiuje kontrakt dla zaawansowanej modyfikacji historii (amend, interaktywny rebase i jego akcje pomocnicze).
* GitBranchMerge_Protocol(Protocol): Definiuje kontrakt dla zarządzania gałęziami i scalania (rozwiązywanie konfliktów, weryfikacja stanu, usuwanie bezpieczne/wymuszone).
* GitTag_Protocol(Protocol): Definiuje kontrakt dla obsługi tagów (lekkie, opisowe, listowanie, usuwanie).
* GitCherryPick_Protocol(Protocol): Definiuje kontrakt dla wybiórczego aplikowania rewizji (cherry-pick pojedynczego commita).
* Zależności i ograniczenia: Wymaga modułu standardowego typing.Protocol. Dziedziczenie po Protocol powoduje strukturalne sprawdzanie zgodności typów (duck typing) przez static type checkery (np. mypy). Klasy nie implementują żadnej logiki biznesowej ani algorytmów (zawierają instrukcje wielokropka ...). Zależy zewnętrznie od struktury importu z modułu a01_Nauka_Aktualna.
* Kontrakt danych (Wejście/Wyjście): Wejście: Brak (definicje abstrakcyjne). Wyjście: Brak (sygnatury metod zwracają typ None).


## KLASA: GitBasic
* Główna odpowiedzialność: Realizacja funkcji edukacyjno-prezentacyjnych wyświetlających standardowe składnie i polecenia systemu kontroli wersji Git dla podstawowych operacji na repozytorium.
* Stan obiektu (Atrybuty self):
* opis_menu: [str] - Ciąg tekstowy definiujący sekwencję słów kluczowych wykorzystywany przez silnik nadrzędny do mapowania i filtrowania interfejsu.
* Szczegółowy opis metod:
* version() -> None`: Wyświetla instrukcję sprawdzania wersji oprogramowania Git w konsoli.
* config_global() -> None`: Prezentuje polecenie konfiguracji globalnej tożsamości użytkownika.
* init() -> None`: Generuje w strumieniu wyjściowym komendę inicjalizacyjną nowego lokalnego repozytorium.
* status() -> None`: Wypisuje instrukcję weryfikacji bieżącego stanu obszaru roboczego i indeksu.
* add() -> None`: Prezentuje składnię polecenia dodającego wszystkie modyfikacje do obszaru przejściowego (staging).
* commit() -> None`: Wyświetla wzorzec operacji tworzenia nowej rewizji z wiadomością tekstową.
* push() -> None`: Wypisuje schemat wypychania lokalnych zatwierdzeń do zdalnego repozytorium na gałąź główną.
* pull() -> None`: Prezentuje komendę pobierania oraz scalania zmian ze zdalnego serwera.
* restore_checkout_switch() -> None`: Generuje sformatowane zestawienie trzech odrębnych instrukcji Git służących do przywracania zawartości plików, przełączania kontekstu rewizji oraz zmiany aktywnych gałęzi.
* Zależności i ograniczenia: Dziedziczy bezpośrednio po klasach __BazaNauki__ oraz GitBasic_Protocol. Poprawne działanie metod zależy od mechanizmu przechwytywania i modyfikowania standardowego strumienia wyjściowego przez obiekt Edukator w wyższych warstwach systemu. Wszystkie metody posiadają ukryty efekt uboczny polegający na bezpośrednim wywoływaniu funkcji print.
* Kontrakt danych (Wejście/Wyjście): Wejście: Brak (metody bezargumentowe wywoływane poprzez mechanizmy refleksji). Wyjście: Brak (bezpośrednia modyfikacja bufora wyjściowego terminala).


## KLASA: GitDiff
* Główna odpowiedzialność: Eksponowanie poleceń oraz technik porównywania zmian (mechanizmu różnicowego) systemu kontroli wersji Git w celach edukacyjnych za pomocą ujednoliconego interfejsu tekstowego.
* Stan obiektu (Atrybuty self):
* opis_menu: [str] - Ciąg tekstowy zawierający listę metod przeznaczony do parsowania i kategoryzacji przez klasę bazową.
* Szczegółowy opis metod:
* o_hash_importance() -> None`: Prezentuje komunikat informujący o roli sum kontrolnych SHA-1 w identyfikacji punktów historycznych repozytorium.
* diff_standard() -> None`: Wyświetla podstawowe polecenie do weryfikacji zmian w plikach niezatwierdzonych.
* diff_stat() -> None`: Prezentuje składnię generującą statystyki ilości zmodyfikowanych wierszy.
* diff_shortstat() -> None`: Wyświetla skrócone do jednej linii podsumowanie modyfikacji.
* diff_patch() -> None`: Pokazuje polecenie generujące szczegółowy format łatki z wymuszeniem kolorowania składni.
* diff_cached() -> None`: Prezentuje komendę porównującą pliki w obszarze staging z ostatnim zatwierdzeniem.
* diff_kumulacja() -> None`: Instruuje o możliwości łączenia parametrów konfiguracyjnych w jednym wywołaniu.
* diff_commits() -> None`: Przedstawia składnię porównującą stan kodu między dwoma konkretnymi rewizjami.
* diff_head_shortcuts() -> None`: Wyświetla nowożytne skróty referencyjne wskaźnika HEAD oraz operatory zakresów.
* diff_head_relative() -> None`: Pokazuje składnię porównań względnych z użyciem przesunięcia o zadaną liczbę rewizji.
* diff_commits_selective() -> None`: Prezentuje polecenie ograniczające analizę różnicową między rewizjami do wybranej ścieżki pliku.
* diff_separator() -> None`: Wyjaśnia rolę i składnię podwójnego myślnika jako separatora oddzielającego referencje od ścieżek dyskowych.
* Zależności i ograniczenia: Dziedziczy funkcjonalność i zachowanie z klas __BazaNauki__ oraz GitDiff_Protocol. Poprawne formatowanie komunikatów zależy od zewnętrznych dekoratorów i metod klasy Edukator. Wywołuje ukryte efekty uboczne poprzez bezpośrednie modyfikowanie strumienia wyjściowego funkcją print.
* Kontrakt danych (Wejście/Wyjście): Wejście: Brak (metody bezargumentowe wywoływane przez mechanizm refleksji). Wyjście: Brak (strumień tekstowy na standardowe wyjście konsoli).


## KLASA: GitLog
* Główna odpowiedzialność: Ekspozycja instruktażowych szablonów poleceń CLI dla systemu kontroli wersji Git, mapowanych automatycznie przez mechanizm refleksji interfejsu edukacyjnego.
* Stan obiektu (Atrybuty self):
* opis_menu: [str] - Ciąg tekstowy definiujący jawną listę metod i kolejność ich prezentacji w menu CLI.
* Klasa dziedziczy stan oraz metody sterujące z klasy bazowej __BazaNauki__ (brak zdefiniowanego własnego konstruktora __init__).
* Szczegółowy opis metod:
* basic_log(): Wyświetla podstawowe komendy przeglądu historii skróconej (--oneline) oraz limitowania liczby commitów.
* file_analysis_log(): Prezentuje polecenia szczegółowej inspekcji zmian w plikach na poziomie patchy (-p) oraz metryk statystycznych (--stat).
* search_log(): Eksponuje składnię filtrowania historii według komunikatów zatwierdzeń (--grep) oraz mechanizm przeszukiwania zawartości kodu (Git Pickaxe -S).
* visual_log(): Generuje szablon pełnego, graficznego drzewa rewizji dla wszystkich gałęzi repozytorium.
* log_all_branches(): Pokazuje komendę agregacji historii ze wszystkich wskaźników referencyjnych (--all).
* log_decorate(): Wyświetla instrukcję wizualizacji metadanych, takich jak tagi oraz aktualna pozycja wskaźnika HEAD.
* config_call_alias_ll(): Demonstruje lokalną konfigurację złożonego aliasu git ll w pliku konfiguracyjnym repozytorium.
* config_alias_global(): Pokazuje składnię tworzenia aliasów globalnych, dostępnych w całym środowisku użytkownika systemowego (--global).
* call_alias_ll(): Instruuje o sposobie uruchomienia zdefiniowanego uprzednio aliasu w powłoce systemowej.
* log_graph(): Prezentuje polecenie rysowania struktury drzewiastej ograniczonej wyłącznie do bieżącej gałęzi.
* Zależności i ograniczenia: Dziedziczenie po __BazaNauki__ i implementacja protokołu GitLog_Protocol. Wywołanie metod skutkuje bezpośrednim wypisaniem danych na standardowe wyjście (skutek uboczny: sys.stdout). Poprawność nawigacji zależy od struktury znakowej docstringa klasy.
* Kontrakt danych (Wejście/Wyjście): Wejście: Brak (wywołanie bezargumentowe metod instancji). Wyjście: Brak (wyjście strumieniowe tekstowe typu stdout).


## KLASA: GitUndo
* Główna odpowiedzialność: Ekspozycja wzorców poleceń CLI przeznaczonych do cofania, odwracania i resetowania stanów oraz modyfikacji w systemie kontroli wersji Git.
* Stan obiektu (Atrybuty self):
* opis_menu: [str] - Ciąg tekstowy definiujący jawną listę metod wywołania oraz sekwencję ich indeksowania w CLI.
   * Klasa dziedziczy mechanizmy stanu z klasy bazowej __BazaNauki__ (brak zdefiniowanego własnego konstruktora __init__).
* Szczegółowy opis metod:
* quick_commit(): Prezentuje polecenie natychmiastowej rejestracji zmian dla wszystkich plików śledzonych (-am).
* revert_changes(): Wyświetla składnię tworzenia bezpiecznego rewersu zatwierdzenia na podstawie podanego identyfikatora commitu.
* revert_no_commit(): Prezentuje polecenie przygotowania zmian odwracających w przestrzeni roboczej bez automatycznego tworzenia rewizji (-n).
* restore_file_from_hash(): Pokazuje instrukcję przywrócenia pliku ze źródła referencyjnego na podstawie hashu (-s).
* reset_soft(): Eksponuje komendę cofnięcia wskaźnika gałęzi z zachowaniem zmian w obszarze przejściowym (Staging Area).
* reset_mixed(): Pokazuje domyślne polecenie cofnięcia wskaźnika rewizji przenoszące modyfikacje do katalogu roboczego (Working Directory).
* reset_hard(): Wyświetla destrukcyjne polecenie całkowitego porzucenia zmian w przestrzeni roboczej i indeksie.
* checkout_file_from_hash(): Prezentuje historyczną składnię selektywnego wypakowania wersji pliku z określonego punktu w czasie.
* Zależności i ograniczenia: Ścisłe dziedziczenie po __BazaNauki__, zgodność strukturalna z protokołem GitUndo_Protocol oraz zależność od interpreterów sekwencji ANSI w systemie operacyjnym hosta.
* Kontrakt danych (Wejście/Wyjście): Wejście: Brak (metody bezargumentowe wywoływane przez refleksję silnika). Wyjście: Brak (strumień danych wyjściowych bezpośrednio na sys.stdout).


## KLASA: GitRefactor
* Główna odpowiedzialność: Ekspozowanie i instruktaż szablonów CLI służących do zaawansowanej modyfikacji historii rewizji, zarządzania procesem rebase oraz naprawiania commitów w systemie Git.
* Stan obiektu (Atrybuty self):
* opis_menu: [str] - Ciąg tekstowy definiujący jawną listę metod refaktoryzacyjnych oraz ich sekwencyjny indeks w interfejsie CLI.
* Klasa dziedziczy mechanizmy stanu z klasy bazowej __BazaNauki__ (brak zdefiniowanego własnego konstruktora __init__).
* Szczegółowy opis metod:
* commit_amend(): Prezentuje polecenie modyfikacji ostatniego commitu poprzez zmianę opisu lub dołączenie nowych zmian bez tworzenia osobnej rewizji.
* rebase_interactive(): Eksponuje składnię inicjalizacji interaktywnego procesu przebudowy historii (-i) dla określonej liczby ostatnich commitów.
* rebase_pick(): Wyświetla opis standardowej instrukcji zachowania wybranego commitu w pliku konfiguracyjnym rebase.
* rebase_reword(): Pokazuje instrukcję zatrzymania procesu w celu edycji wyłącznie treści komunikatu wybranego zatwierdzenia.
* rebase_squash(): Prezentuje polecenie scalenia commitu z poprzedzającym wraz z możliwością modyfikacji połączonego komunikatu.
* rebase_fixup(): Instruuje o sposobie cichego wtopienia zmian z commitu do jego poprzednika bez zachowywania opisu nowszej rewizji.
* rebase_drop(): Pokazuje komendę pominięcia i trwałego usunięcia wybranego punktu historii z drzewa rewizji.
* rebase_main(): Prezentuje procedurę przeniesienia bazy bieżącej gałęzi na wierzchołek gałęzi głównej (main).
* rebase_interactive_head(): Pokazuje składnię dynamicznego definiowania zakresu edycji historii względem aktualnego wskaźnika HEAD.
* rebase_continue(): Wyświetla polecenie wzniesienia i kontynuacji wstrzymanego procesu rebase po manualnym usunięciu konfliktów scalania.
* rebase_abort(): Eksponuje komendę natychmiastowego przerwania operacji rebase i bezpiecznego przywrócenia stanu repozytorium sprzed jej rozpoczęcia.
* Zależności i ograniczenia: Dziedziczenie po klasie bazowej __BazaNauki__ oraz implementacja interfejsu zgodnego z GitRefactor_Protocol. Poprawność wyświetlania zależy od obsługi sekwencji ANSI w terminalu oraz zgodności formatu dokumentacji klasy.
* Kontrakt danych (Wejście/Wyjście): Wejście: Brak (metody bezargumentowe wywoływane mechanizmem refleksji). Wyjście: Brak (efekt uboczny: wypisanie danych formatowanych bezpośrednio na sys.stdout).


## KLASA: GitBranchMerge
* Główna odpowiedzialność: Ekspozycja wzorców instruktażowych CLI przeznaczonych do zarządzania cyklem życia gałęzi repozytorium, operacji scalania oraz procedur rozwiązywania konfliktów w systemie Git.
* Stan obiektu (Atrybuty self):
* opis_menu: [str] - Ciąg tekstowy definiujący jawną listę metod zarządzania gałęziami oraz kolejność ich indeksowania w CLI.
* Klasa dziedziczy architekturę stanu z klasy bazowej __BazaNauki__ (brak zdefiniowanego własnego konstruktora __init__).
* Szczegółowy opis metod:
* merge_standard(): Prezentuje podstawową komendę scalania wskazanej gałęzi bocznej do aktualnie aktywnego wskaźnika HEAD.
* merge_with_msg(): Wyświetla składnię operacji łączenia gałęzi z jawnym nadpisaniem domyślnego komunikatu zatwierdzenia za pomocą flagi -m.
* resolve_conflicts(): Generuje instrukcję opisującą procedurę manualnej eliminacji znaczników konfliktu (<<<<, ====, >>>>) z plików źródłowych.
* branch_move(): Prezentuje polecenie zmiany nazwy istniejącej gałęzi za pomocą operacji przeniesienia referencji (--move).
* branch_list_merged(): Pokazuje komendę filtrującą gałęzie, których modyfikacje zostały w pełni zintegrowane z bieżącym punktem historii (--merged).
* branch_list_unmerged(): Eksponuje polecenie listowania gałęzi zawierających unikalne, niepołączone jeszcze commity (--no-merged).
* branch_delete_safe(): Instruuje o sposobie bezpiecznego usuwania gałęzi, odrzucającego operację w przypadku wykrycia niezłączonych zmian (-d).
* branch_delete_force(): Prezentuje destrukcyjne, wymuszone usuwanie gałęzi niezależnie od statusu integracji jej kodu (-D).
* branch_from_hash(): Wyświetla technikę rekonstrukcji lub inicjalizacji nowej gałęzi bezpośrednio na podstawie wskazanego hashu rewizji.
* Zależności i ograniczenia: Ścisłe dziedziczenie po klasie __BazaNauki__, implementacja kontraktu GitBranchMerge_Protocol oraz zależność wyjściowa od strumienia systemowego sys.stdout.
* Kontrakt danych (Wejście/Wyjście): Wejście: Brak (metody bezargumentowe wywoływane automatycznie przez refleksję silnika). Wyjście: Brak (efekt uboczny: generowanie komunikatów tekstowych bezpośrednio na standardowe wyjście).


## KLASA: GitTag
* Główna odpowiedzialność: Ekspozycja wzorców komend interfejsu wiersza poleceń (CLI) dedykowanych do zarządzania cyklem życia znaczników (tagów) wersji w repozytorium Git.
* Stan obiektu (Atrybuty self):
* opis_menu: [str] - Ciąg znaków definiujący sekwencyjną listę metod obsługi tagów eksponowanych w menu CLI.
* Klasa dziedziczy mechanizmy zarządzania stanem po klasie bazowej __BazaNauki__ (brak zdefiniowanego własnego konstruktora __init__).
* Szczegółowy opis metod:
* tag_lightweight(): Wyświetla składnię tworzenia lekkiego znacznika będącego bezpośrednią referencją wskaźnika do określonego commitu.
* tag_list(): Prezentuje polecenie listujące wszystkie zarejestrowane znaczniki wersji w strukturze projektu.
* tag_annotated(): Eksponuje komendę generowania tagu adnotowanego (tworzącego pełny obiekt bazy danych Git) z wymaganą wiadomością (-a, -m).
* tag_show(): Pokazuje instrukcję inspekcji szczegółowych metadanych znacznika oraz skojarzonego z nim zatwierdzenia zmian.
* tag_delete(): Prezentuje polecenie lokalnego usuwania wybranego znacznika wersji z repozytorium (-d).
* Zależności i ograniczenia: Dziedziczenie po klasie __BazaNauki__ oraz implementacja interfejsu zgodnie z GitTag_Protocol. Wyświetlanie danych tekstowych zależy od poprawnej obsługi kodowania znaków oraz sekwencji ANSI terminala.
* Kontrakt danych (Wejście/Wyjście): Wejście: Brak (metody bezargumentowe wywoływane automatycznie przez refleksję silnika). Wyjście: Brak (efekt uboczny: strumieniowanie danych informacyjnych bezpośrednio na sys.stdout).


## KLASA: GitCherryPick
* Główna odpowiedzialność: Ekspozycja wzorców instruktażowych interfejsu wiersza poleceń (CLI) służących do wybiórczego przenoszenia i aplikowania określonych zmian (commitów) pomiędzy gałęziami w systemie Git.
* Stan obiektu (Atrybuty self):
* opis_menu: [str] - Ciąg znaków definiujący sekwencyjną listę metod operacji cherry-pick eksponowanych w menu CLI.
* Klasa dziedziczy mechanizmy zarządzania stanem po klasie bazowej __BazaNauki__ (brak zdefiniowanego własnego konstruktora __init__).
* Szczegółowy opis metod:
* cherry_pick_single(): Prezentuje polecenie pobrania zmian z pojedynczej, wskazanej na podstawie hashu rewizji i zaaplikowania jej jako nowego zatwierdzenia w bieżącej gałęzi.
* Zależności i ograniczenia: Dziedziczenie po klasie __BazaNauki__ oraz implementacja interfejsu zgodnie z protokołem GitCherryPick_Protocol. Wyświetlanie danych tekstowych zależy od poprawnej obsługi kodowania znaków oraz sekwencji ANSI terminala.
* Kontrakt danych (Wejście/Wyjście): Wejście: Brak (metody bezargumentowe wywoływane automatycznie przez refleksję silnika). Wyjście: Brak (efekt uboczny: strumieniowanie danych informacyjnych bezpośrednio na sys.stdout).


## Instrukcja warunkowa punktu wejścia / Skrypt główny
* Główna odpowiedzialność: Inicjalizacja oraz uruchomienie interaktywnego środowiska edukacyjnego (CLI) poprzez agregację i przekazanie zarejestrowanych modułów lekcyjnych Git do silnika sterującego.
* Stan obiektu (Atrybuty self):
* Brak struktur obiektowych self (kod stanowi główny punkt wejścia programu __main__ uruchamiający instancję klasy zewnętrznej).
* Szczegółowy opis metod:
* __main__: Blok warunkowy wykonujący sekwencyjne mapowanie klas lekcyjnych do kolekcji moje_lekcje oraz inicjalizujący obiekt __BazaNauki__ z flagą aktywacji trybu interaktywnego.
* Zależności i ograniczenia: Jawna zależność od obecności klas logicznych (GitBasic, GitDiff, GitLog, GitUndo, GitRefactor, GitBranchMerge, GitTag, GitCherryPick) oraz silnika bazowego __BazaNauki__. Wymaga terminala obsługującego interaktywne operacje wejścia/wyjścia (I/O).
* Kontrakt danych (Wejście/Wyjście): Wejście: Referencje obiektów klas (lista typów przekazywana do konstruktora). Wyjście: Brak (efekt uboczny: uruchomienie pętli zdarzeń CLI i przejęcie kontroli nad procesem standardowego wejścia/wyjścia).
