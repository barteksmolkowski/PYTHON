import numpy as np
import torch

data = [[1, 2], [3, 4]]
data = torch.tensor(data)
np_data = np.array(data)
torch_on_np = torch.from_numpy(np_data)

torch_zeros = torch.zeros(5, 3, dtype=torch.long)
torch_ones = torch.ones(5, 3, dtype=torch.long)
torch_rand = torch.rand(5, 3)

print(f"torch_on_np: {torch_on_np}")
print(f"torch_zeros: {torch_zeros}")
print(f"torch_zeros.view(dtype=torch.int): {torch_zeros.view(dtype=torch.int)}")

print(f"torch_ones: {torch_ones}")
print(f"torch_rand: {torch_rand}")


from sklearn.preprocessing import StandardScaler

# dowiedzieć się o TensorFlow i JAX (ten JAX chyba umiem ale tak dla pewności)
# TensorFlow o ile będzie potrzebny w ai dev
# from transformers import pipeline

# generator = pipeline("text-generation", model="gpt2")

# result = generator(
#     "Once upon a time",
#     max_new_tokens=100,
#     temperature=0.7,
#     top_k=50,
#     top_p=0.9
# )
# print(result)

matrix = torch.zeros(3, 5, dtype=torch.int16)
print(f"\n\nmatrix:\n{matrix}")
print(f"matrix.shape: {matrix.shape}")
print(f"matrix[0]: {matrix[0]}")

T = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], device="cpu"
)  # device="cpu" domyślne ale mozna ustawić na "cuda" lub inne
T = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True
)  # sluzy do obliczania gradientow czy wstecznej propagacji

device = "cuda" if torch.cuda.is_available() else "cpu"
T = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True
)

print(f"T: {T}")
print(
    f"T.shape: {T.shape} \nT.dtype: {T.dtype} \nT.device: {T.device} \nT.requires_grad: {T.requires_grad}"
)

T1, T2, T3, T4 = (
    torch.empty(size=(1, 2)),
    torch.eye(2, 2),
    torch.arange(start=1, end=3, step=1),
    torch.linspace(start=0.1, end=1, steps=3),
)
print(
    "\n"
    f"torch.empty(size=(1,2)): \n{T1}\n\n"
    f"torch.eye(2, 2): \n{T2}\n\n"
    f"torch.arange(start=1, end=3, step=1): \n{T3}\n\n"
    f"torch.linspace(start=0.1, end=1, steps=3): \n{T4}\n\n"
)

gen = torch.Generator()
gen.manual_seed(42)

T1 = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
T12 = torch.empty(size=(1, 5)).normal_(mean=0, std=1, generator=gen)
T2 = torch.empty(size=(1, 5)).uniform_(1, 5)

print(
    "<===>\ntorch.empty(...)\n"
    "\n=> .normal_(mean: float = 0, std: float = 1, *, generator: Generator | None = None)   # Rozkład normalny\nmean = centrum rozkładu \nstd = odchylenie \n* = wszystko po jako nazwa\ngenerator = None lub mozna własny generator=gen) normal_ - rozkład z odchyleniem"
    "\ngen = torch.Generator() , gen.manual_seed(42)"
    "\n\n=> .uniform_(from_: float = 10, to: float = 1, *, generator: Generator | None = None) # Rozkład jednostajny\nfrom_ = min.liczba\nto = max.liczba\n(... inne wytłumaczone ...))\n"
    f"\nWszystkie przykłady:\n-> normal(mean=0, std=1):"
    + " " * 17
    + f"{T1}\n-> normal_(mean=0, std=1, generator=gen): {T12}\n-> uniform_(1, 5):"
    + " " * 23
    + f" {T2}\n<===>\n"
)

x = torch.tensor([2.0], requires_grad=True)

# forward
y = x**2 + 3 * x

# backward
y.backward()

print("grad:", x.grad)

# reset
if x.grad is not None:
    x.grad.zero_()

# drugi krok
y = x**2 + 3 * x
y.backward()

print("grad:", x.grad)

T = torch.diag(torch.tensor([1, 2, 3]))
print(f"\n\ntorch.diag(torch.ones(size=(1, 3))):\n{T}")

T = torch.arange(5)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(
    f"\ntorch.arange(5): {T}\n"
    f".bool(): {T.bool()}  # 0 → False, reszta → True\n"
    f".short(): {T.short()}  # int16 (ucięcie, nie zaokrągla)\n"
    f".float(): {T.float()}  # float32 (standard w AI)\n"
    f".long(): {T.long()}  # int64 (indeksy, loss, klasy)\n"
    f".device: {T.device}  # gdzie tensor siedzi (CPU/GPU)\n"
    f".dtype: {T.dtype}  # typ danych (np. int64, float32)\n"
    f".to(device): {T.to(device)}  # przeniesienie na CPU/GPU\n"
)  # half, double

np_array = np.zeros((1, 2), dtype=int)
T = torch.from_numpy(np_array)
np_array_back = T.numpy()
print(
    f"np_array = np.zeros((1, 2), dtype=int): np_array = {np_array}\n"
    f"T = torch.from_numpy(np_array): T = {T}\n"
    f"np_array_back = T.numpy(): np_array_back = {np_array_back}"
)


x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z1 = torch.empty(3, dtype=torch.int16)
torch.add(x, y, out=z1)


print(
    f"{z1.tolist()}  .tolist() używać gdy: potrzebujesz zwykłej listy Pythona (np. JSON, zapis do pliku, API, sklearn)"
)

if z1.device.type == "cpu":
    print(
        f"{z1.numpy()}  .numpy() używać gdy: chcesz przejść z PyTorch do NumPy (np. matplotlib, sklearn, analiza danych)"
    )
else:
    print("Tensor jest na GPU → użyj: z1.cpu().numpy()")


x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z1 = torch.empty(3, dtype=torch.int16)

torch.add(x, y, out=z1)
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# =========================
# PODSTAWOWE OPERACJE
# =========================

add = x + y  # dodawanie
sub = x - y  # odejmowanie
mul = x * y  # mnożenie (element-wise)
div = x / y  # dzielenie

# alternatywy funkcji
add2 = torch.add(x, y)
sub2 = torch.sub(x, y)
mul2 = torch.mul(x, y)
div2 = torch.div(x, y)

true_div = torch.true_divide(x, y)  # jak / → zawsze float

# =========================
# FUNKCJE MATEMATYCZNE
# =========================

pow2 = torch.pow(x, 2)  # potęga
sqrt_x = torch.sqrt(x)  # pierwiastek
exp_x = torch.exp(x)  # e^x
log_x = torch.log(x)  # log
abs_x = torch.abs(x)  # |x|

# =========================
# REDUKCJE (STATYSTYKA)
# =========================

sum_x = torch.sum(x)
mean_x = torch.mean(x)
max_x = torch.max(x)
min_x = torch.min(x)

argmax_x = torch.argmax(x)  # indeks największej wartości
argmin_x = torch.argmin(x)

# =========================
# SHAPE / TENSOR OPERACJE
# =========================

shape = x.shape  # rozmiar
view = x.view(3, 1)  # zmiana kształtu
reshape = x.reshape(3, 1)  # bezpieczniejsza wersja view

# =========================
# AUTOGRAD
# =========================

x2 = torch.tensor([2.0], requires_grad=True)

y2 = x2**2 + 3 * x2
y2.backward()

grad = x2.grad  # gradient

# reset gradientu
if x2.grad is not None:
    x2.grad.zero_()

detach = x2.detach()  # odcina gradient

# =========================
# KONWERSJE / WARTOŚCI
# =========================

scalar = x2.item()  # tensor → liczba

# =========================
# BROADCASTING
# =========================

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([1.0])

broadcast = a + b  # [2,3,4]

# =========================
# IN-PLACE (OSTROŻNIE)
# =========================

t = torch.tensor([1.0, 2.0, 3.0])
t.add_(1)  # IN-PLACE: zmienia tensor (t = t + 1)

# =========================
# DODATKOWE PRZYDATNE
# =========================

clamp = torch.clamp(x, min=1.5, max=2.5)  # ogranicza wartości
round_x = torch.round(x)  # zaokrągla
floor_x = torch.floor(x)  # podłoga (w dół)
ceil_x = torch.ceil(x)  # sufit (w górę)


A_1D, B_1D = torch.tensor([1, 2]), torch.tensor([3, 4])
A_2D, B_2D = (
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[3, 4], [5, 6]], dtype=torch.float32),
)

C1 = A_2D.mm(B_2D)  # mnozenie macierzy
C2 = A_2D.matrix_power(3)
C3 = A_1D.dot(B_1D)
print(f"{C1}\n{C2}\n{C3}")


batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))

out_bmm = torch.bmm(tensor1, tensor2)

print("=== SHAPES ===")
print("tensor1 shape:", tensor1.shape)
print("tensor2 shape:", tensor2.shape)
print("out_bmm shape:", out_bmm.shape)

print("\n=== PRZYKŁAD (batch 0) ===")
print("tensor1[0] shape:", tensor1[0].shape)
print("tensor2[0] shape:", tensor2[0].shape)
print("out_bmm[0] shape:", out_bmm[0].shape)

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
torch.sort
print(f"(dim = 0, keepdim = False):\n{torch.sum(x, dim=0, keepdim=False)}\n")
print(f"(dim = 0, keepdim = True):\n{torch.sum(x, dim=0, keepdim=True)}\n")
print(f"(dim = 1, keepdim = False):\n{torch.sum(x, dim=1, keepdim=False)}\n")
print(f"(dim = 1, keepdim = True):\n{torch.sum(x, dim=1, keepdim=True)}\n")

# eq(input, other, *, out)                              # Porównanie element po elemencie (True/False)
#                                                       # -> other - to z czym porownujesz input
# sort(input, *, stable, dim=-1, descending=False, out) # Sortowanie
#                                                       # -> stable - zachowuje kolejność elementów o tej samej wartości
#                                                       # -> descending - sortuje od największego do najmniejszego
# clamp(input, min, max, *, out)                        # Wycinanie wartości poza granicami
#                                                       # -> min - dolna granica (wszystko poniżej = min)
#                                                       # -> max - górna granica (wszystko powyżej = max)

a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 5, 3])

T = torch.eq(a, b)
print(f"porównanie {a} i {b}: {T}")

T = torch.zeros((6, 8), dtype=torch.int16)
print(T[0].shape)
print(T[:, 0].shape)
print(T[2, 1:4].shape)
T[0, 0] = 100
print(T)

T = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(T, T[rows, cols].shape)

X = torch.arange(10)
print(X)
Y1 = X[(X < 2) | (X > 8)]  # | -> OR
Y2 = X[(X < 2) & (X > 8)]  # & -> AND
Y3 = X[X.remainder(2) == 0]  # x.remainder(y) == z -> if x % y == z
Y4 = torch.where(
    X % 2 == 0, X // 2, X * 3 + 1
)  # condition - warunek, input - gdy True, other - gdy False
Y5a, Y5b, Y5c = torch.tensor([1, 1, 2, 3, 3]).unique(
    sorted=True,
    return_inverse=True,  # dla każdego elementu mówi z którego unikalnego indeksu pochodzi
    return_counts=True,  # ile razy każda unikalna wartość występuje
    dim=-1,
)
print(Y5a, Y5b, Y5c)
print(f"\n\n\n{Y1}\n{Y2}\n{Y3}\n{Y4}\n{Y5a}, {Y5b}, {Y5c}")

x = torch.rand(2, 3, 4)  # macierz 2x3x4
print(x.dim(), x.numel())  # dim=3 (3 wymiary), numel=24 (2*3*4 elementy)

T = torch.arange(9)
T3_3 = T.view(3, 3)
print(T3_3)
T3_3 = T.reshape(3, 3)
print(T3_3)

T = torch.arange(6).reshape(2, 3)
print(T)
T = T.t()
print(T)

print("\n\n\n\n")

# start
x = torch.arange(6).view(2, 3)
print("x:\n", x)
print("x.is_contiguous():", x.is_contiguous())  # True → dane są „po kolei”

# transpozycja
y = x.t()  # zamiana wierszy ↔ kolumn
print("\ny = x.t():\n", y)
print("y.is_contiguous():", y.is_contiguous())  # False → dane „rozsypane”

# ❌ view nie działa
try:
    y.view(6)
except:
    print("\ny.view(6) -> ❌ błąd (non-contiguous)")

# ✅ naprawa
y2 = y.contiguous()  # robi kopię i układa dane poprawnie
print("\ny2.is_contiguous():", y2.is_contiguous())

print("y2.view(6):", y2.view(6))  # działa

# =========================
# INNE OPERACJE OSI
# =========================

print("\ntranspose:")
print(x.transpose(0, 1))  # to samo co .t() ale jawnie podajesz osie

print("\npermute:")
z = torch.rand(2, 3, 4)
print("z.shape:", z.shape)

z2 = z.permute(2, 0, 1)
# zmiana kolejności osi -> z.permute(0->2, 1->0, 2->1) i wtedy torch.rand(2, 3, 4) -> torch.rand(4, 2, 3)

print("z2.shape:", z2.shape)

T1 = torch.tensor([[1, 2], [3, 4]])

T2 = torch.tensor([5, 6]).unsqueeze(
    0
)  # torch.tensor([5, 6]).unsqueeze(0): [5, 6] -> [[5, 6], ] dodaje kolejny wymiar

print(torch.cat([T1, T2], dim=0))
