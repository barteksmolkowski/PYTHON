# import jax  # -------------# Używane do ultra-szybkich badań i symulacji (wkrótce)
# import jax.numpy as jnp  # # Odpowiednik NumPy w JAX działający na GPU/TPU (wkrótce)
import numpy as np  # -----# Podstawowa biblioteka do operacji na macierzach na CPU

# import tensorflow as tf  # # Standard produkcyjny w starszych/wielkich systemach (wkrótce)
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import (
    StandardScaler,  # Przygotowanie i normalizacja danych (wkrótce)
)

### 1. MOST NUMPY <-> PYTORCH (Współdzielenie pamięci RAM) ###
data = torch.tensor([[1, 2], [3, 4]])
np_data = np.array(data)
print(f"numpy -> torch: {torch.from_numpy(np_data)}")

### 2. GENEROWANIE I REZERWOWANIE TENSORÓW ###
tensor_zeros = torch.zeros(5, 3, dtype=torch.long)  # Tensor wypełniony fizycznie zerami
tensor_ones = torch.ones(
    5, 3, dtype=torch.long
)  # Tensor wypełniony fizycznie jedynkami
tensor_empty = torch.empty(
    5, 3, dtype=torch.float32
)  # Szybka alokacja; zawiera śmieci z pamięci RAM

torch.rand(
    5, 3, out=tensor_empty
)  # Nadpisuje przygotowaną pamięć losowymi liczbami z (0, 1)
tensor_zeros.view(
    dtype=torch.int
)  # Zmiana sposobu interpretacji typu danych przez widok
print(
    f"Pierwszy wiersz: {tensor_zeros[0]}"
)  # Wyciąganie pojedynczego wiersza przez indeksowanie

### 3. ATRAKCYJNOŚĆ I KONFIGURACJA SPRZĘTOWA TENSORA ###
device = "cuda" if torch.cuda.is_available() else "cpu"

T = torch.tensor(
    [[1, 2, 3], [4, 5, 6]],
    dtype=torch.float32,
    device=device,  # Lokalizacja sprzętowa tensora (CPU / GPU)
    requires_grad=True,  # Włącza śledzenie operacji do wstecznej propagacji (Autograd)
    pin_memory=False,  # True betonuje RAM przy treningu na GPU pod ultraszybki transfer w DataLoaderze
)

print(
    f"T: {T}\nT.shape: {T.shape} (kształt) \nT.dtype: {T.dtype} (typ danych) \n"
    f"T.device: {T.device} (lokalizacja) \nT.requires_grad: {T.requires_grad} (śledzenie)\n"
)

### 4. STRUKTURY LINIOWE I CIĄGI LICZBOWE ###
print(
    f"{torch.eye(2, 2)} -> Macierz jednostkowa (jedynki na głównej przekątnej, reszta zera)\n"
    f"{torch.arange(start=1, end=3, step=1)} -> Generowanie: znasz krok (od start do end-1)\n"
    f"{torch.linspace(start=0.1, end=1, steps=3)} -> Generowanie: znasz ilość oczekiwanych elementów\n"
)

### 5. GENERATORY I ROZKŁADY PRAWDOPODOBIEŃSTWA ###
gen = torch.Generator()
gen.manual_seed(
    42
)  # Blokuje punkt startowy losowości, zapewniając powtarzalność wyników

print(
    f"{torch.empty(size=(1, 5)).normal_(mean=0, std=1)} -> Rozkład normalny (Gaussa) wokół średniej\n"
    f"{torch.empty(size=(1, 5)).normal_(mean=0, std=1, generator=gen)} -> Losowanie zablokowane seedem\n"
    f"{torch.empty(size=(1, 5)).uniform_(1, 5)} -> Rozkład jednostajny (równe szanse w przedziale)\n"
)

######################################################################

x = torch.tensor([[1.0, 2.0]], requires_grad=True)
model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))

"""1. BLOKADA UCZENIA (no_grad) -> Inference / Walidacja / Testy"""

# W `no_grad` chodzi o to, że NOWA operacja na T z requires_grad=True NIE tworzy śladu w grafie (brak grad_fn).
with torch.no_grad():
    y_test = model(x)  # Wynik sieci bez śledzenia operacji (oszczędność pamięci)

    # WIĘCEJ PRZYKŁADÓW: Operacje wykonują się matematycznie, ale ignorują autograd
    z1 = x * 2  # z1.grad_fn to None -> brak śledzenia mnożenia
    z2 = torch.exp(x)  # z2.grad_fn to None -> brak śledzenia funkcji e^x
    z3 = z1 + z2  # z3.grad_fn to None -> złożona operacja też jest czysta
    # WYNIK: Wszystko liczy się szybciej, ale na z1/z2/z3 nie wywołasz .backward()

""" 2. ZAMRAŻANIE WARSTW -> Co dokładnie zawiera i zwraca .parameters()? """

# CO ZWRACA: Generator obiektów typu nn.Parameter (czyli tensorów z wagami i requires_grad=True).
# CO ZAWIERA dla model[0] (nn.Linear):
#   1. model[0].weight -> Tensor wag (macierz połączeń o kształcie [out_features, in_features])
#   2. model[0].bias   -> Tensor przesunięć (wektor o kształcie [out_features])

# PRZYKŁAD 1: Zamrażanie wybranej warstwy (Podstawa Transfer Learningu)
# ZASADA: .parameters() to generator, dlatego pętla `for` to jedyna poprawna metoda w PyTorch.
for param in model[0].parameters():
    param.requires_grad = (
        False  # Zamraża wagi i bias warstwy [0] -> nie będą zbierać gradientów
    )

# PRZYKŁAD 2: Inicjalizacja wag (Drugie najważniejsze zastosowanie w AI dev)
for param in model.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(
            param
        )  # Ręczne nadpisanie wartości startowych wszystkich wag w sieci

""" 3. MAPA OPERACJI (grad_fn) I WSTECZNA PROPAGACJA (.backward) """

# WIZUALIZACJA: grad_fn NIE jest płaską listą. To drzewo binarne (graf). Każdy tensor pamięta
# tylko operację, która go stworzyła oraz skąd przyszły dane poprzez referencje (.next_functions).
y = model(x)

# Jak wygląda historia operacji w pamięci?
# y.grad_fn                  ➔ Zwraca: <AddmmBackward0> (Ostatnia warstwa Linear)
# y.grad_fn.next_functions   ➔ Zwraca: ((<TBackward0>, 0), (<AddmmBackward0>, 0))
#                                        └─ Referencja do wcześniejszej warstwy Linear w głąb sieci!

print(f"Ślad operacji (Ostatni węzeł grafu): {y.grad_fn}")

# PRZYKŁAD: Klasyczna pętla treningowa (Gdzie żyje .backward i .step)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(1):  # W prawdziwym kodzie np. 100 epok
    optimizer.zero_grad()  # 1. Czyszczenie starych notatek (.grad)
    outputs = model(x)  # 2. Forward pass (Przejście danych w przód)
    loss = criterion(outputs, torch.tensor([[1.0]]))  # 3. Obliczenie błędu (Loss)

    loss.backward()  # 4. Wsteczna propagacja (Oblicza pochodne i zapisuje w .grad)
    optimizer.step()  # 5. Aktualizacja wag (Modyfikuje liczby w wagach o wartość .grad)

""" 4. ODCIĘCIE GRADIENTU (.detach) -> Kontekst użycia w kodzie """

# KONTEKST: Używasz, gdy w trakcie treningu chcesz wyciągnąć wynik do innej biblioteki
# (np. wykresy matplotlib, metryki sklearn), ale nie chcesz zapychać pamięci grafem obliczeń.
losses_to_plot = []

for epoch in range(1):
    outputs = model(x)
    loss = criterion(outputs, torch.tensor([[1.0]]))
    loss.backward()

    # KOD BEZPIECZNY: .detach() odcina linię autogradu, a .item() rzuca czysty skalar poza pętlę.
    losses_to_plot.append(loss.detach().item())

    # GDYBYŚ ZROBIŁ: losses_to_plot.append(loss) bez odcięcia, PyTorch trzymałby w RAM-ie
    # całe wielkie drzewa grafów ze wszystkich epok. Po kilku minutach dostaniesz błąd "Out of Memory".

""" 5. ZAAWANSOWANE PARAMETRY METODY .backward() """

# A. gradient=None -> Tensor wag (np. [1.0, 0.5]) mówiący wstecznej propagacji, które błędy są ważniejsze.
# B. retain_graph=True -> Zachowuje graf obliczeń przy liczeniu kilku niezależnych błędów z jednego przejścia danych.
y_shared = model(x)
loss_pies = torch.abs(y_shared - torch.tensor([[1.0]]))
loss_rasa = torch.abs(y_shared - torch.tensor([[0.5]]))

loss_pies.backward(
    retain_graph=True
)  # "Nie pal mapy!" - zachowuje rusztowanie dla drugiego zadania
loss_rasa.backward()  # Liczy błąd rasa i ostatecznie zwalnia graf z pamięci RAM

# C. create_graph=True -> Graf wyższego rzędu (Sam gradient staje się funkcją matematyczną).
# ZASTOSOWANIE: Pozwala policzyć pochodną z pochodnej (druga pochodna, np. przyspieszenie w sieciach fizycznych PINN).
x_phys = torch.tensor([2.0], requires_grad=True)
y_phys = x_phys**3  # Graf: y = x^3
dy_dx = torch.autograd.grad(y_phys, x_phys, create_graph=True)[
    0
]  # Graf pierwszej pochodnej: y' = 3x^2
d2y_dx2 = torch.autograd.grad(dy_dx, x_phys)[
    0
]  # Obliczenie drugiej pochodnej: y'' = 6x

# D. inputs=None -> Pozwala policzyć gradienty (.grad) wyłącznie dla wybranych warstw lub elementów.
loss_custom = torch.abs(model(x) - torch.tensor([[1.0]]))
loss_custom.backward(
    inputs=list(model[1].parameters())
)  # .grad zostanie obliczony i wpisany TYLKO do warstwy [1]


""" CZĘŚĆ 2: REALIZACJA (Kod do odpalenia) """

# A. Budowa modelu i zamrażanie "mózgu" (warstwy [0])
model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))

# Dla każdego parametru (Weights + Bias) w warstwie [0] -> wyłącz naukę
for param in model[0].parameters():
    param.requires_grad = False

x = torch.tensor([[1.0, 2.0]], requires_grad=True)
target_pies = torch.tensor([[1.0]])
target_rasa = torch.tensor([[0.5]])

# B. Tryb: TYLKO PODGLĄD (Blokada)
with torch.no_grad():
    y_test = model(x)
    print(f"1. Podgląd (no_grad) -> grad_fn: {y_test.grad_fn}")  # Wyświetli: None

# C. Tryb: POWRÓT DO NAUKI (Automatyczny po wyjściu z taba)
y = model(x)
print(f"2. Nauka (powrót) -> grad_fn: {y.grad_fn}")  # Wyświetli: <AddmmBackward0>

# D. Liczenie dwóch błędów (Dwa cele)
loss1 = torch.abs(y - target_pies)
loss1.backward(retain_graph=True)  # "Nie pal mapy!"

loss2 = torch.abs(y - target_rasa)
loss2.backward()  # Teraz mapa znika

# E. Odcięcie do statystyk i czyszczenie
wynik_wykres = y.detach().numpy()  # Odcinamy "linę" i idziemy do Numpy
model.zero_grad()
if x.grad is not None:
    x.grad.zero_()

print(f"3. Wynik odcięty (detach): {wynik_wykres} | Gradienty wyczyszczone.")

######################################################################
# Dane bazowe
T = torch.arange(5)
device = "cuda" if torch.cuda.is_available() else "cpu"

### 1. KONWERSJA TYPÓW ORAZ ANALIZA STANU (.dtype / .device) ###
# Sprawdzanie stanu:
current_device, current_dtype = (
    T.device,
    T.dtype,
)  # Sprzęt (CPU/CUDA) oraz aktualny typ danych
T_moved = T.to(device)  # Uniwersalny przerzut na inny sprzęt lub typ

# Metody rzutowania typów (Od najlżejszych do najcięższych):
t_bool = T.bool()  # Zwraca Bool (0: False, reszta: True)
t_int16 = T.short()  # Zwraca int16 (Szybkie operacje na całkowitych)
t_float16 = T.half()  # Zwraca float16 (Drastyczna oszczędność pamięci VRAM na GPU)
t_float32 = T.float()  # Zwraca float32 (Standardowa precyzja w Deep Learningu)
t_float64 = T.double()  # Zwraca float64 (Maksymalna precyzja matematyczna)
t_int64 = T.long()  # Zwraca int64 (Wymagany typ pod indeksowanie, klasy i funkcje Loss)

### 2. EKSPORT I TRANSFER DANYCH NA ZEWNĄTRZ PYTORCHA ###
x, y = torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])
z1 = torch.empty(3, dtype=torch.int16)
torch.add(x, y, out=z1)  # Dodawanie z zapisem bezpośrednio do przygotowanej pamięci out

# Eksport do czystego Pythona (Używaj pod: JSON, API, zapis do plików tekstowych, serializację):
pure_list = z1.tolist()

# Eksport do NumPy (Używaj pod: matplotlib, scikit-learn, zaawansowaną analizę danych):
if z1.device.type == "cpu":
    np_array_back = (
        z1.numpy()
    )  # Działa bezpośrednio na CPU (współdzieli pamięć z tensorem)
else:
    np_array_back = (
        z1.cpu().numpy()
    )  # ZAWSZE zrzuć najpierw na CPU, jeśli tensor leżał na GPU!

### 3. SPECYFICZNE FUNKCJE STRUKTURALNE ###
# torch.diag(wektor) tworzy macierz kwadratową z elementami wektora na głównej przekątnej
T_diag = torch.diag(torch.tensor([1, 2, 3]))

np_array = np.zeros((1, 2), dtype=int)
T_from_np = torch.from_numpy(
    np_array
)  # Bezpieczny most NumPy -> PyTorch (współdzielona pamięć)

################################################################################################
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

""" 1. PODSTAWOWE (Element-wise) -> Działania pozycjami; alternatywa: torch.add/sub/mul/div """

add, sub = x + y, x - y
mul, div = x * y, x / y

""" 2. FUNKCJE MATEMATYCZNE -> Operacje na każdym elemencie z osobna """

pow2, sqrt_x = x**2, torch.sqrt(x)
exp_x, log_x = torch.exp(x), torch.log(x)
abs_x = torch.abs(x)  # Wartość bezwzględna

""" 3. REDUKCJE (Statystyka globalna) -> Agregacja całego tensora do jednej liczby """

sum_x, mean_x = torch.sum(x), torch.mean(x)
max_x, min_x = torch.max(x), torch.min(x)

""" 4. POZYCJE EKSTREMÓW (Arg-funkcje) -> Wyciąganie indeksów zamiast wartości """

argmax_x = torch.argmax(x)  # idx poz., max wartosc
argmin_x = torch.argmin(x)  # idx poz., min wartosc

################################################################################################
# Dane bazowe
x = torch.tensor([1.0, 2.0, 3.0])
A_1D, B_1D = torch.tensor([1, 2]), torch.tensor([3, 4])
A_2D = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
B_2D = torch.tensor([[3, 4], [5, 6]], dtype=torch.float32)

### 1. ROZMIAR I ZMIANA KSZTAŁTU (Pamięć ciągła vs nieciągła) ###
shape_x = x.shape  # Rozmiar i wymiary tensora
v_x = x.view(3, 1)  # Zmiana kształtu; wywali błąd, jeśli dane w RAM nie są po kolei
r_x = x.reshape(
    3, 1
)  # Bezpieczna zmiana kształtu; sam zrobi kopię, jeśli dane są rozsypane

### 2. AUTOGRAD (Ręczny minikrok gradientu) ###
x2 = torch.tensor(
    [2.0], requires_grad=True
)  # Włącza śledzenie operacji do wstecznej propagacji
y2 = x2**2 + 3 * x2
y2.backward()  # Liczy pochodną (2x + 3) i wpisuje wynik do x2.grad

if x2.grad is not None:
    x2.grad.zero_()  # Ręczny reset (zerowanie) gradientu przed kolejnym obliczeniem

x2_detached = (
    x2.detach()
)  # Odcina tensor od grafu; służy do bezpiecznego wyciągania danych do wykresów
scalar = x2.item()  # Konwersja jednoelementowego tensora na zwykłą liczbę w Pythonie

### 3. BROADCASTING (Rozciąganie wymiarów) ORAZ OPERACJE IN-PLACE ###
b1 = torch.tensor([1.0, 2.0, 3.0]) + torch.tensor(
    [1.0]
)  # Dodanie 1D + 0D -> rozszerza [1.0] do [1.0, 1.0, 1.0]
b2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) + torch.tensor(
    [3, 2, 1]
)  # Dodanie 2D + 1D wierszami

t = torch.tensor([1.0, 2.0, 3.0])
t.add_(1)  # Modyfikacja IN-PLACE (podkreślenie _): nadpisuje ten sam obszar pamięci
t = t.add(1)  # Zwykła modyfikacja: alokuje nowy, osobny tensor w pamięci RAM

### 4. PRZYCINANIE, ZAOKRĄGLENIA I MATEMATYKA MACIERZY ###
clamp = torch.clamp(
    x, min=1.5, max=2.5
)  # Ścina wartości, zamykając je w sztywnych widłach min/max
rnd, flr, cel = (
    torch.round(x),
    torch.floor(x),
    torch.ceil(x),
)  # Zaokrąglenia: standardowe, w dół, w górę

C1 = A_2D.mm(B_2D)  # Klasyczne mnożenie macierzy 2D (wiersze x kolumny)
C2 = A_2D.matrix_power(3)  # Podnoszenie macierzy kwadratowej do potęgi algebraicznej
C3 = A_1D.dot(B_1D)  # Iloczyn skalarny dwóch płaskich wektorów 1D

################################################################################################

# 1. BMM ORAZ AGREGACJE OSI (Batch Matrix Multiplication i .sum) ###
# bmm: mnożenie macierzy 3D w paczkach (B, N, M) @ (B, M, P) -> (B, N, P)
# Przydatne przy jednoczesnym przepychaniu całego batcha danych przez model.
t1, t2 = torch.rand((32, 10, 20)), torch.rand((32, 20, 30))
out_bmm = torch.bmm(t1, t2)  # Wynikowy kształt: [32, 10, 30]

x_sum = torch.tensor([[1, 2, 3], [4, 5, 6]])
# dim: 0 = po kolumnach, 1 = po wierszach | keepdim: True zachowuje oryginalną liczbę osi (wymiar)
s0_flat = torch.sum(
    x_sum, dim=0, keepdim=False
)  # Wynik: płaski wektor 1D -> [5, 7, 9], kształt [3]
s0_keep = torch.sum(
    x_sum, dim=0, keepdim=True
)  # Wynik: zachowana macierz 2D -> [[5, 7, 9]], kształt [1, 3]
s1_flat = torch.sum(
    x_sum, dim=1, keepdim=False
)  # Wynik: płaski wektor 1D -> [6, 15], kształt [2]
s1_keep = torch.sum(
    x_sum, dim=1, keepdim=True
)  # Wynik: zachowana macierz 2D -> [[6], [15]], kształt [2, 1]

### 2. LOGIKA, PORÓWNANIA I WYCINANIE (Slicing / Indexing) ###
# torch.eq(): element-wise porównanie; zwraca maskę logiczną typu Bool
mask_eq = torch.eq(
    torch.tensor([1, 2, 3]), torch.tensor([1, 5, 3])
)  # Wynik: [True, False, True]

T_slice = torch.zeros((6, 8), dtype=torch.int16)
row0, col0, block = (
    T_slice[0],
    T_slice[:, 0],
    T_slice[2, 1:4],
)  # Wycinanie osiami (wiersz, kolumna, wycinek wiersza)
T_slice[0, 0] = 100  # Bezpośrednia modyfikacja konkretnego elementu w macierzy

# Advanced Indexing (Fancy Indexing) przy użyciu list/tensorów z indeksami (wyciąga punkty [1,4] oraz [0,0]):
T_rand = torch.rand((3, 5))
fancy_indexed = T_rand[torch.tensor([1, 0]), torch.tensor([4, 0])]

# 3. MASKA WARUNKOWA, FILTROWANIE I UNIKATY ###
X = torch.arange(10)
Y1 = X[
    (X < 2) | (X > 8)
]  # Bitowy operator | (OR): elementy mniejsze od 2 LUB większe od 8
Y2 = X[(X < 2) & (X > 8)]  # Bitowy operator & (AND): łączny (tutaj da pusty tensor)
Y3 = X[X.remainder(2) == 0]  # .remainder() to odpowiednik % (wybiera liczby parzyste)

# torch.where(warunek, gdy_true, gdy_false) -> wektorowy odpowiednik if-else
Y4 = torch.where(X % 2 == 0, X // 2, X * 3 + 1)

# .unique() wyciąga unikatowe wartości; zwraca (wartości, indeksy rekonstrukcji, licznik wystąpień)
unique_vals, inverse_idx, counts = torch.tensor([1, 1, 2, 3, 3]).unique(
    return_inverse=True, return_counts=True
)

# 4. KONTROLA PARAMETRÓW STRUKTURY I KSZTAŁTU ###
x_struct = torch.rand(2, 3, 4)
axes_count, total_elements = (
    x_struct.dim(),
    x_struct.numel(),
)  # .dim() = osie (3) | .numel() = łączna liczba liczb (24)

# view wymaga ciągłości w RAM, reshape to bezpieczny automat kopiujący w razie potrzeby:
T_base = torch.arange(9)
T_v, T_r = T_base.view(3, 3), T_base.reshape(3, 3)

T_trans = (
    torch.arange(6).reshape(2, 3).t()
)  # .t() szybko transponuje macierz 2D (zamiana osi 0 i 1)

################################################################################################
### 1. CIĄGŁOŚĆ PAMIĘCI (Contiguous vs Non-contiguous) ###
x = torch.arange(6).view(2, 3)  # x.is_contiguous() == True (dane leżą w RAM po kolei)
y = x.t()  # Transpozycja niszczy ciągłość: y.is_contiguous() == False

# y.view(6) wywali błąd! Naprawa wymaga .contiguous():
y2 = y.contiguous()  # Tworzy nową kopię i układa dane po kolei w pamięci RAM
final_view = y2.view(6)  # Teraz zmiana kształtu przez .view() działa poprawnie

### 2. MANIPULACJE OSIAMI (Transpose, Permute, Unsqueeze, Cat) ###
trans_x = x.transpose(
    0, 1
)  # Zamiana dwóch konkretnych osi miejscami (dla 2D to samo co .t())

z = torch.rand(2, 3, 4)
z2 = z.permute(
    2, 0, 1
)  # Globalne przestawienie osi: zmienia kształt z (2, 3, 4) na (4, 2, 3)

T1 = torch.tensor([[1, 2], [3, 4]])
T2 = torch.tensor([5, 6]).unsqueeze(
    0
)  # Wstrzykuje wymiar o rozmiarze 1 na pozycji 0: z [5, 6] robi [[5, 6]]

cat_result = torch.cat(
    [T1, T2], dim=0
)  # Skleja tensory w jeden blok wzdłuż podanej osi dim (tutaj wierszami)
