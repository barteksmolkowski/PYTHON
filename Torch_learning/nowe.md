Nauka O - nie zrobione, 1 - zrobione. Kazdy dział to szacunkowo 5h dzialania
- PyTorch:
    ### Dzień 1: Refaktoryzacja teorii i bazy warstw
    01) X Upewnić się, że dopisano już wszystkie nn.layers – nie wszystkie, tylko te najbardziej przydatne (Linear, Embedding, LayerNorm, BatchNorm2d, Dropout, ReLU, GELU, SiLU).
    02) X Dopisać pozostałe parametry do wszystkich nn.layers.
    03) X Podzielić warstwy na kategorie semantyczne przy użyciu promptu nr 2 (Core, Standard, Adaptive, Output, Specialized itd.).
    04) X Przenieść powtórki parametrów na samą górę danej kategorii w formie zwięzłego podsumowania.
    05) S Uzupełnić treść w polach 'notatka: ' dla wszystkich sklasyfikowanych warstw.
    06) S Zorganizować opisy specyficznych parametrów bezpośrednio pod nazwą każdej warstwy, zapewniając czytelność.

    ### Dzień 2: Czyszczenie kodu i zaawansowane hiperparametry
    07) X Poprawić kolory w edytorze lub składni notatek, aby były rzeczywiste i widoczne, a nie ukryte wewnątrz docstringów.
    08) O Zweryfikować rozdział o zaawansowanych hiperparametrach – usunąć teorie nieprzydatne w pracy komercyjnej, zostawiając tylko te kluczowe do nauki.
    09) O Usunąć zbędne opisy parametrów bezpośrednio z wnętrza definicji klasy 'class NN1(nn.Module)'.
    10) X Uporządkować cały plik tak, aby IDE nie podświetlało błędów na czerwono (błędy składni/importów), a ostrzeżenia z AI lądowały jako żółte uwagi.
    11) X Usunąć sekcję 'WARSTWY UCZĄCE SIĘ' aż do sekcji 'warstwa forward', opierając się wyłącznie na sekcji 'OGÓLNE ZASADY ATRYBUTÓW'.

    ### Dzień 3: Budowa architektur sieci neuronowych
    12) O Usunąć wszystkie pozostałe, nadmiarowe definicje klas class(nn.Module) – pozostawić w pliku tylko 1 wymaxowaną klasę (MLP) oraz 2 puste miejsce na nową strukturę.
    13) O Zbudować w przygotowanej drugiej klasie od zera działający mechanizm Mini-Transformera (Self-Attention, warstwy liniowe, projekcje).
    14) O Poprawić formatowanie tekstu i kodu wewnątrz metody 'forward' (usunąć szare, nieaktywne bloki kodu, zostawiając czysty i czytelny zapis).

    ### Dzień 4: Przygotowanie danych i inicjalizacja
    15) O Sekcja 'Load Data' – napisać pobieranie zbioru (np. MNIST) oraz konfigurację obiektów Dataset i DataLoader (batch_size, shuffle).
    16) O Sekcja 'Initialize network' – stworzyć instancje przygotowanych wcześniej klas modeli (MLP oraz Mini-Transformer) z odpowiednimi wymiarami wejściowymi.
    17) O Sekcja 'Loss and optimizer' – zdefiniować funkcję straty (np. CrossEntropyLoss) oraz podpiąć optymalizator (np. Adam) pod parametry sieci.

    ### Dzień 5: Pętla treningowa i ewaluacja
    18) O Sekcja 'Trainer Network' – napisać kompletną pętlę uczącą po epokach zawierającą zerowanie gradientów, forward pass, obliczenie straty, backward pass oraz krok optymalizatora.
    19) O Sekcja 'Check accuracy on training & test to see how good our model' – stworzyć funkcję sprawdzającą dokładność modelu w trybie ewaluacji bez obliczania gradientów (torch.no_grad()).

- PyTorch from scratch
    ###
    Złóż swoje dotychczasowe klocki NumPy w kompletną, prostą sieć wielowarstwową (MLP).
    Wytrenuj ją na prostym zbiorze (np. MNIST lub danych syntetycznych).
    Zapisz wagi sieci do pliku (np. za pomocą pickle lub np.save), aby móc je potem załadować.

- Scikit-Learn
    ### Przetwarzanie i inżynieria danych
    01) Zrozumieć, dlaczego modele ML nie akceptują tekstu i wymagają liczb.
    02) Poznać pojęcie "wycieku danych" (Data Leakage) – dlaczego skalujemy dane dopiero po ich podziale.
    03) Zrozumieć różnicę między cechami (Features / X) a etykietą docelową (Target / y).
    04) Wczytać dowolny zbiór danych tablicowych za pomocą biblioteki Pandas.
    05) Podzielić dane na cechy wejściowe (X) oraz etykiety docelowe (y).
    06) Podzielić zbiór na część treningową i testową za pomocą train_test_split (np. test_size=0.2, random_state=42).
    07) Dopasować i przetransformować cechy liczbowe za pomocą StandardScaler.
    08) Przekształcić kolumny kategoryczne (tekstowe) na postać binarną za pomocą OneHotEncoder.
    
    ### Modelowanie i metryki
    09) Zrozumieć intuicję stojącą za Regresją Logistyczną (klasyczny model liniowy) a Lasem Losowym (model oparty na drzewach decyzyjnych).
    10) Zrozumieć "Macierz Pomyłek" (Confusion Matrix) i pojęcia: True Positive, False Positive, True Negative, False Negative.
    11) Dowiedzieć się, dlaczego Accuracy (dokładność) potrafi oszukiwać przy niezbalansowanych zbiorach danych i kiedy patrzeć na Precision/Recall.
    12) Zainicjalizować i wytrenować model LogisticRegression przy użyciu metody .fit(X_train, y_train).
    13) Zainicjalizować i wytrenować model RandomForestClassifier na tych samych danych.
    14) Wygenerować predykcje dla obu modeli na zbiorze testowym za pomocą metody .predict(X_test).
    15) Zaimportować z sklearn.metrics i obliczyć dla obu modeli metrykę Accuracy (ogólna dokładność).
    16) Obliczyć i porównać metryki Precision (precyzja) oraz Recall (czułość).
    17) Obliczyć metrykę F1-Score (średnia harmoniczna) i przeanalizować, który model poradził sobie lepiej.

- FastAPI
    ### Podstawy web api & architektura
    01) Zrozumieć protokół HTTP oraz różnicę między metodą GET (pobieranie) a POST (wysyłanie danych).
    02) Stworzyć bazową aplikację FastAPI i napisać pierwszy asynchroniczny endpoint GET ("/") zwracający status serwera (tzw. Health Check).
    03) Opanować podstawy asynchroniczności (kiedy używać słów kluczowych `async/await`, a kiedy blokujących funkcji synchronicznych).

    ### Walidacja i integracja modelu
    04) Zrozumieć rolę biblioteki Pydantic w walidacji danych (jak zapobiegać awariom serwera, gdy użytkownik wyśle błędny format danych).
    05) Napisać klasę Pydantic (BaseModel), która ściśle definiuje strukturę i typy danych przesyłanych do Twojego modelu (np. lista floatów).
    06) Zaimplementować mechanizm ładowania wag Twojego modelu NumPy ("from scratch") dokładnie w momencie startu aplikacji FastAPI.
    07) Stworzyć endpoint POST ("/predict"), który przyjmuje przwalidowane dane od użytkownika, przepuszcza je przez logikę sieci NumPy i zwraca wynik w formacie JSON.

    ### Obsługa błędów i prowadzenie inferencji
    08) Zrozumieć kody statusu HTTP (200 OK, 400 Bad Request, 422 Unprocessable Entity, 500 Internal Server Error).
    09) Zabezpieczyć endpoint predict za pomocą bloku `try/except` i obiektu `HTTPException` – serwer musi bezpiecznie obsłużyć błędy matematyczne lub niepoprawne wymiary macierzy.
    10) Przeprowadzić testy API przy użyciu wbudowanej w FastAPI automatycznej dokumentacji Swagger UI (`/docs`), wysyłając poprawne oraz celowo uszkodzone zapytania.

- Docker
    ### Podstawy konteneryzacji i konfiguracja
    01) Zrozumieć różnicę między obrazem (szablonem) a kontenerem (działającą instancją) oraz dlaczego Docker eliminuje problem "u mnie nie działa".
    02) Zainstalować Docker Desktop na swoim systemie operacyjnym.
    03) Przetestować poprawność instalacji środowiska, uruchamiając w terminalu testowy obraz komendą `docker run hello-world`.

    ### Tworzenie środowiska dla aplikacji
    04) Dowiedzieć się, jak działają warstwy w Dockerze i jak optymalnie układać instrukcje w pliku konfiguracyjnym.
    05) Napisać plik `Dockerfile` w głównym katalogu swojej aplikacji FastAPI (wybrać lekki obraz bazowy Pythona, ustawić katalog roboczy, skopiować pliki projektu).
    06) Stworzyć plik `requirements.txt` ze wszystkimi bibliotekami (FastAPI, Uvicorn, NumPy, torch itd.) i dodać do Dockerfile krok instalacyjny `pip install`.
    07) Zdefiniować w Dockerfile domyślną komendę uruchomieniową (`CMD`), która podnosi serwer Uvicorn na odpowiednim porcie i hoście (0.0.0.0).

    ### Budowanie i uruchamianie kontenerów
    08) Zrozumieć mechanizm przekierowania portów (port forwarding) między systemem operacyjnym a wnętrzem kontenera.
    09) Zbudować własny obraz produkcyjny za pomocą komendy `docker build -t ai-service .`.
    10) Uruchomić skonteneryzowaną aplikację komendą `docker run -p 8000:8000 ai-service` i przetestować w przeglądarce, czy API działa prawidłowo wewnątrz izolowanego kontenera.

- Nowoczesne GenAI (Embeddings + RAG od zera)
    ### Teoria i generowanie embeddings
    01) Zrozumieć pojęcie osadzeń semantycznych (embeddings) – jak modele językowe zamieniają znaczenie słów i zdań na gęste wektory liczb.
    02) Zainstalować bibliotekę `sentence-transformers` i pobrać lekki, darmowy model z Hugging Face (np. z rodziny MiniLM).
    03) Napisać skrypt wczytujący surowy tekst z dokumentów, dzielący go na mniejsze fragmenty (chunks) i generujący dla każdego z nich wektor znaczeniowy.

    ### Matematyczna wyszukiwarka w NumPy
    04) Zrozumieć matematyczną intuicję stojącą za podobieństwem cosinusowym (Cosine Similarity) jako miarą zbieżności dwóch wektorów.
    05) Zaimplementować wzór na podobieństwo cosinusowe w czystym NumPy bez gotowych bibliotek ML.
    06) Napisać algorytm wyszukiwarki (Retriever), który za pomocą iloczynu skalarnego (`np.dot`) porównuje wektor pytania użytkownika z całą macierzą dokumentów, a następnie wyciąga fragment o najwyższym współczynniku podobieństwa.

    ### Orkiestracja i integracja z zewnętrznym LLM
    07) Zrozumieć mechanizm prompt engineeringu w architekturze RAG – jak wstrzykiwać odnaleziony kontekst do instrukcji dla modelu językowego.
    08) Założyć darmowe konto na Hugging Face, pobrać token API lub zainstalować lokalnie narzędzie Ollama z darmowym modelem (np. Llama 3 / Mistral).
    09) Stworzyć skrypt orkiestrujący: pobrać wyciągnięty przez wyszukiwarkę fragment tekstu, dokleić go jako kontekst do zapytania użytkownika i wysłać kompletny prompt do modelu LLM, odbierając wygenerowaną odpowiedź.

    ### Wdrożenie systemu jako usługa sieciowa
    10) Zintegrować napisaną logikę RAG ze strukturą swojego projektu z Tygodnia 2.
    11) Dodać do serwera FastAPI nowy endpoint POST (`/rag-chat`), który przyjmuje pytanie użytkownika, wykonuje cały cykl RAG i zwraca ostateczną odpowiedź w formacie JSON.


Co zawiera PyTorch from Scratch:
torchFromScratch.md

# 20k znaków kodu, streszcznie -> 100 lini tekstu
CEL (WHAT TO DO): ZADANIE: Przeanalizuj kod projektu ML/Backend i wygeneruj architektoniczne podsumowanie umożliwiające rekonstrukcję systemu. 2. FILTR INFORMACJI (WHAT TO IGNORE / FOCUS): PRIORYTET INFORMACJI: Uwzględniaj tylko elementy wpływające na: execution flow, architekturę systemu, ML/data pipeline, API boundaries, orchestration, state management, performance. IGNORUJ: boilerplate, importy, proste utility functions, CRUD bez logiki, powtarzalne wrappery 3. DEKOMPOZYCJA SYSTEMU (CORE UNDERSTANDING): Zidentyfikuj: entrypoint systemu, centralne klasy i moduły, flow inicjalizacji, zależności między komponentami, wzorzec architektoniczny (jeśli istnieje) 4. ANALIZA WARSTW (SYSTEM BREAKDOWN): (Podziel system na: architecture layer (moduły, struktura), ML/data layer (modele, pipeline, transformacje), API/service layer (endpointy, orchestration), state/lifecycle layer (session, model state)) 5. DATAFLOW (MUST BE STRICT): (Opisz dokładnie: input (format, typ, shape), transformacje pośrednie, operacje macierzowe / ML logic, output (format, struktura)) 6. IMPLEMENTATION CORE: (Wypisz: kluczowe klasy, kluczowe metody, sygnatury API, krytyczne flagi konfiguracji) 7. QUALITY / RISK ANALYSIS: (Wskaż: TODO / not implemented, placeholder logic, dead code, niespójności API, brak walidacji, ryzyka architektoniczne, fragmenty eksperymentalne) 8. OUTPUT RULES: (Styl: maksymalna gęstość informacji, brak wyjaśnień edukacyjnych, brak lania wody, tylko technical notes senior-level, format listowy / strukturalny). DODATKOWE WYMAGANIA: Zachowaj strukturalny format 4-sekcyjny z limitami linii (nie przekraczaj ich). Każda sekcja musi być kompletna i niezależna logicznie. W każdej sekcji wyraźnie oznacz: ENTRYPOINT → FLOW → OUTPUT (jeśli dotyczy). Eksplitycznie oddziel: ARCHITECTURE vs ML LOGIC vs API LAYER vs STATE. W dataflow zawsze podawaj: input shape/type → transformacje → output schema. W każdej analizie dodaj minimalny dependency graph (moduł A → B → C). Jeśli coś jest niepewne, oznacz jako [INFERRED], nie zgaduj. Priorytet: execution flow > dataflow > orchestration > implementation details. DODATKOWE WYMAGANIA GLOBALNE (OBOWIĄZKOWE): (Zachowaj dokładnie 5 sekcji wyjścia: (1–5), bez zmiany kolejności i nazw. Każda sekcja MUSI mieć limit linii: 1: max 10 linii, 2: max 15 linii, 3: max 25 linii, 4: max 15 linii, 5: max 15 linii. Nie scalaj sekcji i nie przenoś treści między nimi. Sekcja 2 MUSI być jawnie oznaczona jako "ZASTOSOWANE TECHNIKI I MATEMATYKA". Sekcja 3 MUSI zawierać pełne sygnatury kluczowych klas i metod (API-level detail). Sekcja 4 MUSI mieć twardą strukturę: input → transformacje → output (bez wyjątków). Każda sekcja MUSI być maksymalnie gęsta informacyjnie (no filler). Jeśli brakuje danych → wpisz [INFERRED], nie pomijaj sekcji. Nie zmieniaj nazw sekcji, nawet jeśli treść sugeruje inaczej. Całość ma być spójna jak jeden raport techniczny (single system document).). FINAL OUTPUT TEMPLATE (STRICT - FOLLOW EXACTLY): ( 1. (ARCHITECTURE\n- ENTRYPOINT:\n- FLOW:\n- OUTPUT:\n) 2. (ZASTOSOWANE TECHNIKI I MATEMATYKA\n- ENTRYPOINT:\n- FLOW:\n- OUTPUT:\n) 3. (KLUCZOWE KOMPONENTY (CLASSES / METHODS / API)\n- ENTRYPOINT:\n- FLOW:\n- OUTPUT:\n) 4. (DATAFLOW\n- INPUT:\n- TRANSFORMATIONS:\n- OUTPUT:\n) 5. (RISK / QUALITY ANALYSIS\n- ENTRYPOINT:\n- FLOW:\n- OUTPUT:\n). RULE: (NIE ZMIENIAJ FORMATU, NIE DODAWAJ DODATKOWYCH SEKCJI, NIE PRZESUWAJ TREŚCI MIĘDZY SEKCJAMI))
.

# SKRACACZ PARU SEKCJI TYCH SAMYCH W JEDNĄ:
ZADANIE: (Scal wiele sekcji tego samego typu (np. wiele sekcji „1. (ARCHITECTURE”) pochodzących z różnych modułów projektu ML/backend) w jedną globalną, spójną sekcję nadrzędną reprezentującą final reconstructed system architecture.) CEL: (Nie twórz podsumowania per-moduł. Nie wykonuj merge tekstowego. Zrekonstruuj jeden unified runtime system obejmujący wszystkie dostarczone fragmenty.) TRYB PRACY: (Dostaniesz wiele fragmentów tego samego numeru sekcji: np. wiele „1. ARCHITECTURE” albo wiele „2. ZASTOSOWANE TECHNIKI I MATEMATYKA” albo wiele „3. KLUCZOWE KOMPONENTY” itd.) Masz: (wykryć overlap semantyczny, usunąć redundancję, zunifikować terminology, odtworzyć execution graph, połączyć dependency graph, zachować ML/runtime semantics, zachować tensor/dataflow continuity, zachować intended architecture nawet jeśli implementation jest stubbed.) NIE RÓB: (nie generuj osobnych podsekcji per moduł, nie streszczaj każdego fragmentu osobno, nie powtarzaj tych samych informacji, nie zachowuj lokalnych opisów jeśli istnieje bardziej globalna wersja, nie upraszczaj execution flow, nie gub runtime order, nie usuwaj risków jeśli dotyczą różnych warstw, nie traktuj wejścia jako wielu systemów.) PRIORYTETY MERGE: (1. execution flow 2. orchestration/runtime 3. state ownership 4. tensor/dataflow continuity 5. ML semantics 6. API boundaries/contracts 7. architectural inconsistencies 8. implementation details) KRYTYCZNE ZASADY SEMANTYCZNE: (Traktuj wszystkie wejściowe sekcje jako fragmenty jednego runtime systemu. Scalaj semantycznie, nie tekstowo. Wynik ma reprezentować reconstructed architecture systemu, a nie merge dokumentów. Jeśli wiele sekcji opisuje ten sam runtime etap: utwórz jeden unified execution stage. Jeśli komponent pojawia się wielokrotnie: opisz go raz, w najbardziej kompletnej i globalnej formie. Zachowuj architecture intent nawet jeśli runtime execution jest niekompletny.) RUNTIME RECONSTRUCTION RULES: (Odtwórz pełny end-to-end runtime pipeline. Łącz lokalne pipeline’y w jeden execution graph. Zachowaj kolejność: ingestion → preprocessing → feature extraction → model execution → forward pass → loss → backward → optimization → orchestration → evaluation. Priorytet ma real runtime flow, nie organizacja plików/modułów. Rekonstruuj: control flow, tensor flow, gradient flow, state propagation, orchestration boundaries.) CANONICALIZATION RULES: (Używaj jednej nazwy dla tego samego konceptu w całym output. Nie duplikuj: (* Protocol + ABC descriptions, * tensor type definitions, * state descriptions, * optimizer semantics, * layer semantics, * mathematical formulas, * repeated architectural patterns.). Jeśli istnieje: lokalna wersja vs globalna wersja: zawsze wybieraj globalną.) ABSTRACTION CONTROL: (Nie mieszaj poziomów abstrakcji w jednym bullet poincie. Oddzielaj: (* orchestration, * tensor flow, * mathematical logic, * API contracts, * runtime state, * execution semantics.). Preferuj runtime semantics nad class listings. Preferuj execution graph nad file/module structure.) GRAPH RECONSTRUCTION: (Buduj jeden unified dependency graph całego systemu. Łącz: Data Pipeline → Feature Extraction → NN Layers → Sequential Model → Trainer → Loss → Optimizer → Evaluation. Pokazuj tylko zależności wpływające na: runtime execution, tensor propagation, gradient propagation, lifecycle/state. Ignoruj dependency noise i utility-level dependencies.) INCOMPLETE SYSTEM HANDLING: (Jeśli system jest stubbed: oddziel: intended architecture vs implemented behavior., Oznaczaj: [INTENDED], [PARTIALLY IMPLEMENTED], [STUBBED], [NON-FUNCTIONAL], [EXECUTION STUB], [API-ONLY], [INCONSISTENT API], [INFERRED], Zachowuj wszystkie krytyczne missing links execution graph.) ANTI-REDUNDANCY RULES: (Zakazane: * ponowne definiowanie tych samych tensor flows, * powtarzanie tych samych ryzyk, * wielokrotne opisywanie tych samych optimizerów/layers, * wielokrotne opisywanie Protocol/ABC hybrid, * lokalne dependency graph jeśli istnieje globalny. Scalaj overlapping risks w unified system risk surface.) DATAFLOW RULES: (Jeśli sekcja dotyczy DATAFLOW: obowiązkowo zachowaj: INPUT → TRANSFORMATIONS → OUTPUT), (Zachowuj: tensor shapes, ndarray dimensions, batch semantics, gradient shapes, forward/backward flow continuity.), (Jeśli shape/type niepewny: oznacz [INFERRED].), (Zachowuj: vectorized ops, normalization, batching, convolution semantics, reshape semantics, optimizer update semantics.) OUTPUT OPTIMIZATION: (Każde zdanie musi wnosić nową informację. Każdy bullet ma rozszerzać reconstruction graph. Usuń wszystkie informacje niewpływające na: runtime, execution, gradients, tensor flow, orchestration, state, ML semantics, API contracts. Maksymalizuj information density per line.) SYSTEM RECONSTRUCTION TARGET: (Finalny output ma wyglądać jak: reverse-engineered architecture document, runtime reconstruction blueprint, internal ML framework audit, execution/runtime design spec, system reconstruction report, a nie jak merge notatek.) OUTPUT RULES: (Zwróć WYŁĄCZNIE jedną scaloną sekcję. Zachowaj dokładnie ten sam nagłówek sekcji co wejście. Nie dodawaj nowych sekcji. Nie dodawaj komentarzy. Nie używaj markdown tables. Zachowaj bullet-point structure. Zachowaj execution order. Zachowaj maksymalną gęstość informacji. Nie generuj fillerów. Nie generuj wyjaśnień edukacyjnych.) STRUKTURA OUTPUT: (Zachowaj dokładnie strukturę sekcji wejściowej. Jeśli wejście to: 1. ARCHITECTURE: (output musi być wyłącznie jedną sekcją). Jeśli wejście to: 4. DATAFLOW: output musi zawierać:* INPUT:\n* TRANSFORMATIONS:\n* OUTPUT:\n.) MERGE STRATEGY: (lok. klasy → unified architecture, lok. flow → global runtime graph, lok. API → unified contract layer, lok. state → unified lifecycle/state model, lok. risks → centralized system risk surface, lok. tensor flows → unified tensor propagation model, lok. execution semantics → end-to-end runtime reconstruction)
.