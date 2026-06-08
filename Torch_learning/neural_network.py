import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

"""
1)X znalesc jakie sa podobne polaczyc je i najlepiej wszystkie wlasnie pogrupowac ze zrobic dwa prompty ten ktory tworzy nazwy grup (prompt 3.1)
    a drugi ktory bedzie rozdzielał (prompt 3.2)
1)  W "OGÓLNE ZASADY ATRYBUTÓW NN" zbierac te wspolne atrybuty warstw nad kategoriami dla kazdej odzielnie   
3)  sprawdzic pisownie atrybutow (czy nn.layer.xyz czy model.xyz itp.) i prawdopodobnie będą pełne nazwy zamiast skrotow a jak beda grupy
    to w nazwie rozdzialu oznaczyc ze jest tak jak jest i skrocic bo rozdzialy wazniejsze niz to jakie oznaczenie
4)  dopisac wytlumaczenia dla atrybutów

"""

# 1 # PROMPT, Liste atrybutow dzieli na grupy i wypisuje te grupy
"""Dostajesz: (listę atrybutów / metod PyTorch). ZADANIE: (Pogrupuj je semantycznie w logiczne kategorie (mental model), NIE wypisuj żadnych opisów, NIE wypisuj atrybutów w środku, ZWRÓĆ TYLKO NAZWY GRUP (nagłówki)). ZASADY: (Grupy mają być maksymalnie ogólne i profesjonalne, unikaj nadmiaru kategorii (preferuj 4-8 grup), usuń duplikaty mentalne (np. cuda()/to() → jedna grupa DEVICE), jeśli coś jest rzadkie → wrzuć do "DEBUG / RZADSZE", jeśli coś jest core API → "CORE API"). FORMAT WYJŚCIA: (Tylko lista nagłówków grup, bez punktów, bez opisów, bez kodu)
"""

# 2 # PROMPT, Wpisuje atrybuty do listy grup
"""Dostajesz: (listę atrybutów / metod PyTorch, listę grup (nagłówków)), ZADANIE: (przypisz KAŻDY atrybut do jednej grupy, NIE twórz nowych grup, NIE zmieniaj nazw grup, usuń duplikaty (jeśli coś pasuje do wielu → wybierz najbardziej "core"), ZASADY: (zachowaj senior-level mental model (production ML / PyTorch), preferuj grupę najbardziej ogólną i użyteczną, jeśli coś jest rzadkie → wrzuć do "DEBUG / RZADSZE" jeśli istnieje, nie dodawaj opisów, tylko struktura), FORMAT WYJŚCIA: (### NAZWA GRUPY\n- atrybut\n- atrybut\n- atrybut ... dalej to inne nazwy grup). |
lista grup: (
1. "CONFIGURATION (MODEL HYPERPARAMETERS)", 2. "TENSORS (WEIGHTS / GRADIENTS / TYPES)", 3. "BUFFER STATE (BATCHNORM / RUNNING STATS)", 4. "MODEL STATE & IO (SAVE / LOAD / DEVICE / TRANSFORM)", 5. "TRAINING MODE (TRAIN / EVAL / FLAGS)", 6. "INTROSPECTION / DEBUG (MODEL STRUCTURE)", 7. "ITERATION / PARAMETER ACCESS (MODEL LOOPS)", 8. "FREEZE / TRAIN CONTROL (requires_grad)", 9. "LAYER-SPECIFIC INTERNALS (RARE / EDGE CASES)", 
)"""

# 3 # PROMPT, Słuzy do: Uzupelnia pisowne atrybutow i opisow
"""Dostajesz: listę atrybutów / metod / parametrów PyTorch, opcjonalnie istniejące grupy / sekcje | ZADANIE: 1. Popraw pełną składnię API: określ czy zapis powinien być: model.xyz, layer.xyz, tensor.xyz, layer.weight.xyz, nn.Module.xyz, nn.Conv2d(...) itd., używaj realnej modern PyTorch składni| 2. Rozwiń niejasne skróty: np. BN → BatchNorm, IO → Input / Output, params → parameters, grads → gradients, używaj pełnych profesjonalnych nazw jeśli poprawiają czytelność| 3. Skracaj nazwy sekcji tylko wtedy gdy: sekcja pozostaje jednoznaczna, zachowany jest senior-level mental model, skrót poprawia czytelność cheat sheeta| 4. Priorytet: poprawny kontekst API > długość nazwy, nazwy sekcji mają być krótkie, atrybuty/metody mają być precyzyjne| 5. Usuń: błędną składnię, pseudo-API, niepoprawne skróty, duplikaty, stare / legacy nazwy | ZASADY: zachowaj styl technical cheat sheet, bez tłumaczenia zmian, bez emoji, bez długich opisów, modern PyTorch only, production/research oriented, senior-level readability | FORMAT WYJŚCIA: ### NAZWA SEKCJI ###, poprawny.atrybut(), poprawny.atrybut, tensor.xyz, layer.weight.xyz
"""

# 4 # PROMPT, Poprawia i uzupelnia pisownie atrybutow w warstwach sieci i opisow
"""PROMPT: „UPROSZCZENIE SYGNATUR PyTorch”. Twoim zadaniem jest przekształcanie sygnatur funkcji/klas z PyTorch (torch.nn) do uproszczonego formatu. Zasady transformacji: (Usuń wszystkie adnotacje typów (: int, : _size_any_t, | None, itp.)., Usuń opisy typów argumentów i zwracanych wartości.), Zachowaj tylko: (nazwę funkcji/klasy, nazwy argumentów, wartości domyślne (ale bez typów), Jeśli argument ma wartość domyślną → zapisz jako arg=value, Jeśli argument nie ma wartości domyślnej → zostaw tylko nazwę argumentu, Usuń cudzysłowy dokumentacyjne (''' ... ''')., Każdą sygnaturę zamień do jednej linii.), DODATKOWA REGUŁA (ważna): (Jeśli argument nie ma jawnej wartości domyślnej, ale jest typem wymaganym, nadal zostaw tylko nazwę argumentu bez typu.)
"""


"""
ta lista: 
1. "CONFIGURATION (MODEL HYPERPARAMETERS)",
2. "TENSORS (WEIGHTS / GRADIENTS / TYPES)",
3. "BUFFER STATE (BATCHNORM / RUNNING STATS)",
4. "MODEL STATE & IO (SAVE / LOAD / DEVICE / TRANSFORM)",
5. "TRAINING MODE (TRAIN / EVAL / FLAGS)",
6. "INTROSPECTION / DEBUG (MODEL STRUCTURE)",
7. "ITERATION / PARAMETER ACCESS (MODEL LOOPS)",
8. "FREEZE / TRAIN CONTROL (requires_grad)",
9. "LAYER-SPECIFIC INTERNALS (RARE / EDGE CASES)", 

"""

"""
ZNACZENIE WYMIARÓW

B = batch size | C = channels / features   | H = height | W = width
N = number of features (wektor po flatten) | seq = sequence length (długość sekwencji, np. tekst)
L = length sygnału (dane 1D, np. audio)    | T = time steps (liczba kroków czasowych / klatek)

CNN (B, C, H, W) - Dane przestrzenne (obrazy). Konwolucje wykrywają lokalne wzorce (krawędzie, tekstury, obiekty) przez przesuwające się filtry.
Przykłady: (32, 1, 28, 28) → MNIST, (32, 3, 224, 224) → RGB, (16, 64, 13, 13) → feature maps

FLATTEN (CNN → MLP) - Spłaszczenie usuwa strukturę przestrzenną i zamienia mapy cech na wektor.
(B, C, H, W) → (B, N) | np: (32, 8, 13, 13) → 1352, (64, 64, 7, 7) → 3136

MLP (napisz rozwiniecie skrotu i tez po polsku co to znaczy) - Model pracuje na wektorach cech i uczy się zależności między wartościami numerycznymi.
(B, N) | np: (32, 10) → dane tablicowe, (32, 1352) → po CNN + flatten, (32, 64) → reprezentacja ukryta

NLP (sekwencje) - Tokeny są mapowane na embeddingi i analizowane jako sekwencja zależności.
(B, seq) → (B, seq, C) | np: (32, 50) → tokeny, (32, 50, 128) → embeddingi

AUDIO (sygnał 1D) - Model analizuje sygnał czasowy lub jego reprezentację (np. spektrogram).
(B, C, L) | np: (32, 1, 16000) → waveform, (32, 64, 500) → features

VIDEO (czas + przestrzeń) - Sekwencja obrazów analizowana jednocześnie w czasie i przestrzeni.
(B, T, C, H, W) | np: (8, 16, 3, 224, 224), (4, 32, 1, 64, 64)

MAPA ZASTOSOWAŃ
CNN lok. wzorce w przestrz -> (B, C, H, W) -> ?
FLATTEN (spłaszczenie)     -> (B, C, H, W) → (B, N)
MLP (decyz. na wekt. cech) -> (B, N) -> ?
NLP (zależności w sekwen.) -> (B, seq) → (B, seq, C)
AUDIO (sygnał czasowy)     -> (B, C, L) -> ?
VIDEO (czas + obraz)       -> (B, T, C, H, W)
"""

# Tworzenie atrybutów dla analizowanej warstwy AI:
"""Przeanalizuj warstwę w PyTorch. Stwórz ściągę atrybutów, które wywołuje się po kropce (np. self.warstwa.xyz), stosując poniższe zasady: 1) Usuń parametry konfiguracyjne, które podaje się wewnątrz nawiasów podczas inicjalizacji. 2) Dodaj atrybuty stanu i statystyk, które najczęściej sprawdza się w trakcie działania programu lub debugowania. 3) Opisz każdy atrybut w 1 linijce według wzorca:# nazwa (krótki opis): self.layer.atrybut -> co zwraca/robi, typ (statystyka/parametr uczony/stan) 4) Skup się na praktyce: wypisz tylko to, co programista faktycznie sprawdza.
"""


# Wymienione: nn.: nn.Linear, nn.Conv2d, nn.ReLU, nn.Sigmoid, nn.Softmax, nn.Dropout, nn.BatchNorm2d, nn.Flatten, nn.MaxPool2d, nn.AvgPool2d


""" lista nr.1
in_features, out_channels, kernel_size, p, groups, pooling, stride | parametry uczone (.weight / .bias): (shape, device, dtype, grad, requires_grad, data)
BatchNorm: (running_mean, running_var, num_batches_tracked), 
INSTRUKCJE (bez .attr): (inplace, ceil_mode, padding_mode)
nn.layer.(weight/bias): (.shape, .device, .grad, .dtype, .data, .requires_grad, .numel(), .is_cuda, .T)
nn.layer: (.training, .state_dict(), .load_state_dict(), .apply(fn), .to(device).cuda() / .cpu())
INTROSPEKCJA Debug Struktury layer = nn.Linear(10, 5) | layer.: (_get_name(), extra_repr()(print(model)), get_extra_state())
DODATKOWE (Rzadsze, ale przydatne): (.transposed, .output_padding, .num_features, .train(), .eval())
Uzywane w petlach for: (.requires_grad_(bool), .named_parameters())
"""

### GLOBALNE ZASADY NAZEWNICTWA ###
# Warstwy co mają 2d są tez jako 1d/2d/3d, wyjątek Linear bo płaski wektor.

### SZCZEGÓŁOWE ZASTOSOWANIE ATRYBUTÓW TENSORA ###
### Warstwy z (.weight/.bias) jako 2d tez jako 1d/2d/3d bez Linear bo płaski wektor.
# Liniowe: nn.Linear | Splotowe: nn.Conv2d, nn.ConvTranspose2d | Normalizujące: nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm2d
# Słownikowe: nn.Embedding (UWAGA: posiada tylko .weight)


### OGÓLNE ZASADY ATRYBUTÓW NN ####################################################################
"""
# WARSTWY #
1. LINEAR / MLP
2. CONVOLUTION
3. POOLING
4. NORMALIZATION
5. ACTIVATIONS
6. REGULARIZATION
7. SEQUENCE / RNN
8. TRANSFORMER / ATTENTION
9. EMBEDDINGS
10. RESHAPE / TENSOR OPS
11. LOSS FUNCTIONS
12. CONTAINER / MODEL STRUCTURE
13. UPSAMPLING
14. SPECIALIZED

DO KATEGORII DODAC WIECEJ WARSTW SIECI NEURONOWEJ, prompt zawiera liste, jednolistosc warstw na tle rozdzialow.
wpisuje sie nazwe kategorii i wypisuje wszystkie warstwy jakie pasuja do tej jak część jest taka ze zmienia sie tylko 1d/2d/3d to oznaczamy jako Xd
(to teraz robic)


1) dopisac pozostale parametry do wszystkich
2) podzielic na kategorie prompt nr.2
3) powtorki parametrow w kategoriach dac na gore kategorii w podsumowaniu jak w innych
4) 

"""  # PROMPT 1: Wpisuje się nn.layers lub cos innego i sprawdza do jakiej nalezy kategorii
"""
ZADANIE: (Twoim zadaniem jest klasyfikowanie warstw PyTorch (torch.nn) do jednej z podanych kategorii semantycznych.). ZASADY: (1. Każda warstwa może należeć tylko do JEDNEJ kategorii 2. Klasyfikuj po funkcji warstwy, NIE po parametrach 3. Jeśli warstwa ma warianty 1D/2D/3D → używaj oznaczenia Xd 4. Jeśli warstwa pasuje do wielu kategorii: (wybierz jej główne zastosowanie) 5. Nie twórz nowych kategorii 6. Modern PyTorch only). KATEGORIE: (1. LINEAR / MLP: (fully connected layers, nn.Linear) 2. CONVOLUTION: (lokalna ekstrakcja cech, nn.ConvXd, nn.ConvTransposeXd) 3. POOLING: (downsampling / spatial reduction, MaxPoolXd, AvgPoolXd, AdaptivePool, FractionalPool, Unpool, LPPool) 4. NORMALIZATION: (stabilizacja aktywacji i gradientów, BatchNormXd, LayerNorm, InstanceNormXd, GroupNorm) 5. ACTIVATIONS: (funkcje nieliniowe, ReLU, GELU, SiLU, Sigmoid, Tanh, LeakyReLU) 6. REGULARIZATION: (redukcja overfittingu, Dropout, DropoutXd) 7. SEQUENCE / RNN: (przetwarzanie sekwencji, RNN, GRU, LSTM) 8. TRANSFORMER / ATTENTION: (mechanizmy uwagi i transformery, MultiheadAttention, Transformer, EncoderLayer, DecoderLayer) 9. EMBEDDINGS: (mapowanie indeksów na wektory, Embedding, EmbeddingBag) 10. RESHAPE / TENSOR OPS: (zmiana kształtu tensorów, Flatten, Unflatten, Identity) 11. LOSS FUNCTIONS: (funkcje straty, CrossEntropyLoss, MSELoss, BCEWithLogitsLoss) 12. CONTAINER / MODEL STRUCTURE: (organizacja modelu, ModuleList, ModuleDict, Sequential) 13. UPSAMPLING: (zwiększanie rozdzielczości, Upsample, PixelShuffle) 14. SPECIALIZED: (niszowe / specjalne operacje, Bilinear, Fold, Unfold)). FORMAT WYJŚCIA: (CATEGORY_NAME)
.

"""  # PROMPT 2: Dzieli grupy na kategorie w grupie
"""
ZADANIE: (Twoim zadaniem jest podzielenie warstw PyTorch (torch.nn) wewnątrz jednej głównej kategorii na logiczne podkategorie semantyczne.). ZASADY: (1. Podkategorie mają opisywać różnice funkcjonalne 2. NIE dziel po parametrach 3. Zachowaj production / research mental model 4. Preferuj: (3-4 podkategorie, czasem 2 jeśli kategoria jest mała) 5. Używaj profesjonalnych nazw: (Core, Standard, Adaptive, Output, Specialized, Efficient, Sequence, Spatial, Container, itd.) 6. Nie twórz sztucznych grup 7. Jeśli warstwa ma warianty 1D/2D/3D: (używaj oznaczenia Xd) 8. Modern PyTorch only 9. Grupuj według głównego zastosowania warstwy 10. Jeśli warstwa jest niszowa / rzadka: (wrzuć do "Specialized")). FORMAT WYJŚCIA: (X. CATEGORY_NAME\n\n### SUBCATEGORY_NAME\nnn.LayerA()\nnn.LayerB()\nnn.LayerC()\n\n### SUBCATEGORY_NAME\nnn.LayerA()\nnn.LayerB()\n\n### SUBCATEGORY_NAME\nnn.LayerA()\nnn.LayerB()\nnn.LayerC()\n. PRZYKŁAD STYLU: (5. ACTIVATIONS\n\n### Core / Standard\nnn.ReLU()\nnn.LeakyReLU()\nnn.GELU()\nnn.SiLU()\n\n### Probabilistic / Output\nnn.Sigmoid()\nnn.Tanh()\nnn.Softmax()\nnn.LogSoftmax()\n\n### Smooth / Self-Regularizing\nnn.Softplus()\nnn.Mish()\nnn.Softsign()\n\n### Efficient / Mobile\nnn.Hardswish()\nnn.Hardtanh()\n)
.

"""  # PROMPT 3: Wklej skonczoną kategorie z miejscami na "notatka: " zeby uzupelnic
"""
ZADANIE:
Refaktoryzacja notatek PyTorch (torch.nn / normalization / layers / API) do ultra-zwięzłego technical-reference formatu.
FORMAT:
nn.Layer(args)
notatka: [1 krótka zwarta linia opisująca działanie, mechanizm, wpływ parametrów i ewentualny status warstwy]
ZASADY:
1. Maksymalnie 1-2 krótkie zdania
2. Styl techniczny, production-ready
3. Bez PURPOSE / MECHANISM / INFLUENCE / RELATION
4. Bez emoji, komentarzy, dygresji
5. Zachowaj pełną semantykę warstwy
6. Dla normalization zawsze uwzględnij:
   - typ statystyki
   - zakres działania
   - wpływ eps/momentum/affine/track_running_stats
7. Jeśli warstwa jest standardem → dopisz krótko
8. Jeśli istnieje nowoczesny zamiennik → dopisz krótko
9. Nie powtarzaj oczywistych rzeczy wynikających z nazwy warstwy
10. Maksymalna gęstość informacji przy minimalnej długości.
PRZYKŁAD:
nn.BatchNormXd(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
notatka: batch-based normalizacja per-channel na statystykach batcha; eps stabilizuje variance, momentum aktualizuje running stats, affine dodaje learnable scale/bias; standard CNN
.


1. CONFIGURATION (MODEL HYPERPARAMETERS)
### Core:
out_features, bias=True, device=None, dtype=None
nn.Linear(in_features)
nn.Bilinear(in1_features, in2_features)
### Variants:
nn.LazyLinear(out_features, bias=True, device=None, dtype=None)


2. CONVOLUTION
### Core:
in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None
nn.Conv2d()
nn.ConvTranspose2d(output_padding=0)

### Lazy variants:
out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, dilation=1, device=None, dtype=None
nn.LazyConv2d()
nn.LazyConvTranspose2d(output_padding=0)


3. POOLING dopisac tylo ze notatka
### Core:
kernel_size, stride=None, padding=0, ceil_mode=False
nn.MaxPoolXd(dilation=1, return_indices=False)
nn.AvgPoolXd(count_include_pad=True, divisor_override=None)
### Adaptive:
nn.AdaptiveMaxPoolXd(output_size, return_indices=False)
### Fractional:
nn.FractionalMaxPool2d(kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)
### Specialized:
nn.LPPoolXd(norm_type, kernel_size, stride=None)


4. NORMALIZATION
dodac: nn.FusedBatchNorm

### batch-based normalization (dataset / batch statistics) ###
num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None

nn.BatchNormXd() 
nn.SyncBatchNorm(process_group=None) 

### Feature / layer normalization (sample-wise) ###

nn.LayerNorm(normalized_shape, eps=0.00001, elementwise_affine=True, bias=True) 

### Channel / group normalization ###
eps=0.00001, device=None, dtype=None

nn.InstanceNorm2d(num_features, momentum=0.1, affine=False, track_running_stats=False) 
nn.GroupNorm(num_groups, num_channels, affine=True) 

### Specialized normalization ###

nn.LocalResponseNorm(size, alpha=1e-4, beta=0.75, k=2.0) 
RMSNorm(hidden_size, eps=0.000001) (from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm)


5. ACTIVATIONS

### Core / standard ###
nn.ReLU(inplace=False)
nn.LeakyReLU(negative_slope=0.01, inplace=False)
nn.GELU(approximate="none")
nn.SiLU(inplace=False)
nn.CELU(alpha=1.0, inplace=True)
nn.ELU(alpha=1, inplace=False)
nn.SELU(inplace=False)

### Probabilistic / output ###
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=None)
nn.LogSoftmax(dim=None)

### Smooth / self-regularizing ###
nn.Softplus(beta=1, threshold=20)
nn.Mish(inplace=False)
nn.Softsign()
nn.Softshrink(lambd=0.5)
nn.Tanhshrink()

### Efficient / mobile / quantized ###
nn.Hardswish(inplace=False)
nn.Hardtanh(min_val=-1, max_val=1, inplace=False)
nn.Hardsigmoid(inplace=True)
nn.Hardshrink(lambd=0.5)


6. REGULARIZATION
p=0.5, inplace=False)
nn.Dropout()
nn.DropoutXd()
nn.AlphaDropout()
nn.FeatureAlphaDropout()


7. SEQUENCE / RNN
device=None, dtype=None, input_size, hidden_size, num_layers=1, bias=True, dropout=0, bidirectional=False, batch_first=False
nn.RNN(nonlinearity="tanh")
nn.GRU()
nn.LSTM(proj_size=0)


# 8. TRANSFORMER / ATTENTION

### Core Attention ###
nn.MultiheadAttention(embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)

### Encoder / Decoder Layers ###
nn.TransformerEncoderLayer(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)
nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=0.00001, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)

### Stacked Containers ###
nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
nn.TransformerEncoder(encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True)

### Full Architecture ###
nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation=F.relu, custom_encoder=None, custom_decoder=None, layer_norm_eps=0.00001, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)


9. EMBEDDINGS
num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None
nn.Embedding(_freeze=False)
nn.EmbeddingBag(mode="mean", include_last_offset=False)


10. RESHAPE / TENSOR OPS
nn.Flatten(start_dim=1, end_dim=-1)
nn.Unflatten(dim, unflattened_size)
nn.Identity(...)


11. LOSS FUNCTIONS

### Classification losses ###
nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean", label_smoothing=0)
nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction="mean", pos_weight=None)

### Regression losses ###
nn.MSELoss(size_average=None, reduce=None, reduction="mean")


12. CONTAINER / MODEL STRUCTURE
modules=None
nn.ModuleList()
nn.ModuleDict()


13. UPSAMPLING
nn.Upsample(size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None)
nn.PixelShuffle(upscale_factor)
nn.MaxUnpoolXd(kernel_size, stride=None, padding=0)


14. SPECIALIZED
### Feature Interaction
nn.Bilinear(in1_features, in2_features, out_features, device=None, dtype=None)

### Spatial Tokenization / Reconstruction
kernel_size, dilation=1, padding=0, stride=1
nn.Fold(output_size)
nn.Unfold()


TENSORS (WEIGHTS / GRADIENTS / TYPES)
.weight
.bias
.shape
.device
.dtype
.grad
.requires_grad
.numel()
.T

BUFFER STATE (BATCHNORM / RUNNING STATS)
running_mean
running_var
num_batches_tracked

Dostajesz: listę atrybutów / metod / parametrów PyTorch, opcjonalnie istniejące grupy / sekcje | ZADANIE: 1. Popraw pełną składnię API: określ czy zapis powinien być: model.xyz, layer.xyz, tensor.xyz, layer.weight.xyz, nn.Module.xyz, nn.Conv2d(...) itd., używaj realnej modern PyTorch składni| 2. Rozwiń niejasne skróty: np. BN → BatchNorm, IO → Input / Output, params → parameters, grads → gradients, używaj pełnych profesjonalnych nazw jeśli poprawiają czytelność| 3. Skracaj nazwy sekcji tylko wtedy gdy: sekcja pozostaje jednoznaczna, zachowany jest senior-level mental model, skrót poprawia czytelność cheat sheeta| 4. Priorytet: poprawny kontekst API > długość nazwy, nazwy sekcji mają być krótkie, atrybuty/metody mają być precyzyjne| 5. Usuń: błędną składnię, pseudo-API, niepoprawne skróty, duplikaty, stare / legacy nazwy | ZASADY: zachowaj styl technical cheat sheet, bez tłumaczenia zmian, bez emoji, bez długich opisów, modern PyTorch only, production/research oriented, senior-level readability | FORMAT WYJŚCIA: ### NAZWA SEKCJI ###, poprawny.atrybut(), poprawny.atrybut, tensor.xyz, layer.weight.xyz
.

MODEL STATE & IO (SAVE / LOAD / DEVICE / TRANSFORM)
.state_dict()
.load_state_dict()
.to(device)
.apply(fn)

TRAINING MODE (TRAIN / EVAL / FLAGS)
.train()
.eval()
.training

ITERATION / PARAMETER ACCESS (MODEL LOOPS)
.parameters()
.named_parameters()
.named_modules()

Dostajesz: listę atrybutów / metod / parametrów PyTorch, opcjonalnie istniejące grupy / sekcje | ZADANIE: 1. Popraw pełną składnię API: określ czy zapis powinien być: model.xyz, layer.xyz, tensor.xyz, layer.weight.xyz, nn.Module.xyz, nn.Conv2d(...) itd., używaj realnej modern PyTorch składni| 2. Rozwiń niejasne skróty: np. BN → BatchNorm, IO → Input / Output, params → parameters, grads → gradients, używaj pełnych profesjonalnych nazw jeśli poprawiają czytelność| 3. Skracaj nazwy sekcji tylko wtedy gdy: sekcja pozostaje jednoznaczna, zachowany jest senior-level mental model, skrót poprawia czytelność cheat sheeta| 4. Priorytet: poprawny kontekst API > długość nazwy, nazwy sekcji mają być krótkie, atrybuty/metody mają być precyzyjne| 5. Usuń: błędną składnię, pseudo-API, niepoprawne skróty, duplikaty, stare / legacy nazwy | ZASADY: zachowaj styl technical cheat sheet, bez tłumaczenia zmian, bez emoji, bez długich opisów, modern PyTorch only, production/research oriented, senior-level readability | FORMAT WYJŚCIA: ### NAZWA SEKCJI ###, poprawny.atrybut(), poprawny.atrybut, tensor.xyz, layer.weight.xyz
.

FREEZE / TRAIN CONTROL (requires_grad)
.requires_grad_(bool)
param.requires_grad = False

INTROSPECTION / DEBUG (MODEL STRUCTURE)
print(model)

LAYER-SPECIFIC INTERNALS (RARE / EDGE CASES)
.num_features
.transposed
.output_padding

Dostajesz: listę atrybutów / metod / parametrów PyTorch, opcjonalnie istniejące grupy / sekcje | ZADANIE: 1. Popraw pełną składnię API: określ czy zapis powinien być: model.xyz, layer.xyz, tensor.xyz, layer.weight.xyz, nn.Module.xyz, nn.Conv2d(...) itd., używaj realnej modern PyTorch składni| 2. Rozwiń niejasne skróty: np. BN → BatchNorm, IO → Input / Output, params → parameters, grads → gradients, używaj pełnych profesjonalnych nazw jeśli poprawiają czytelność| 3. Skracaj nazwy sekcji tylko wtedy gdy: sekcja pozostaje jednoznaczna, zachowany jest senior-level mental model, skrót poprawia czytelność cheat sheeta| 4. Priorytet: poprawny kontekst API > długość nazwy, nazwy sekcji mają być krótkie, atrybuty/metody mają być precyzyjne| 5. Usuń: błędną składnię, pseudo-API, niepoprawne skróty, duplikaty, stare / legacy nazwy | ZASADY: zachowaj styl technical cheat sheet, bez tłumaczenia zmian, bez emoji, bez długich opisów, modern PyTorch only, production/research oriented, senior-level readability | FORMAT WYJŚCIA: ### NAZWA SEKCJI ###, poprawny.atrybut(), poprawny.atrybut, tensor.xyz, layer.weight.xyz
.

"""


class Przyklad(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=10,
        hidden_size=128,
        expansion=1,
        act_fn=nn.ReLU,
        dropout_p=0.3,
        use_cat=False,
    ):
        super().__init__()

        # KONFIGURACJA / PARAMETRY
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.use_cat = use_cat

        # WARSTWY
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                groups=1,
            ),
            nn.BatchNorm2d(num_features=32),  # running_mean / running_var
            act_fn(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                ceil_mode=False,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(32 * 13 * 13, hidden_size),
            nn.Dropout(p=dropout_p),
            act_fn(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: Tensor):
        # FLOW TENSORA
        x = self.features(x)
        x = self.classifier(x)

        return x


model = Przyklad()

# TRYB
model.train()
model.eval()

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# INTROSPEKCJA
print(model)
for k, v in list(model.state_dict().items()):
    print(k, v.shape)

layer = nn.Linear(10, 5)

layer._get_name()
layer.extra_repr()
layer.get_extra_state()

# PARAMETRY / DEBUG
LW, LB = layer.weight, layer.bias
print(LW.shape, LW.device, LW.dtype, LW.grad, LW.requires_grad, LW.numel(), LW.is_cuda)
print(
    LB.shape,
    LB.device,
    LB.dtype,
    LB.grad,
    LB.requires_grad,
    LB.numel(),
    LB.is_cuda,
    LB.T,
)

# PETLE
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    print(name, param.shape)

"""in_features, out_channels, kernel_size, p, groups, pooling, stride | parametry uczone (.weight / .bias): (shape, device, dtype, grad, requires_grad, data)
BatchNorm: (running_mean, running_var, num_batches_tracked), 
INSTRUKCJE (bez .attr): (inplace, ceil_mode, padding_mode)
nn.layer.(weight/bias): (.shape, .device, .grad, .dtype, .data, .requires_grad, .numel(), .is_cuda, .T)
nn.layer: (.training, .state_dict(), .load_state_dict(), .apply(fn), .to(device).cuda() / .cpu())
INTROSPEKCJA Debug Struktury layer = nn.Linear(10, 5) | layer.: (_get_name(), extra_repr()(print(model)), get_extra_state())
DODATKOWE (Rzadsze, ale przydatne): (.transposed, .output_padding, .num_features, .train(), .eval())
Uzywane w petlach for: (.requires_grad_(bool), .named_parameters())"""


# USUNAC PRINTY I HASZTSGI I WDROZYC TE Z GORY W KLASEß

####### Prompt do wypisywania atrybutow #######
"""
Przeanalizuj warstwę PyTorch.
Wypisz sytuacje, w których zasada 'wpisuję xyz w nawiasie, odczytuję jako .xyz' zawodzi, oraz 3 (lub tylko te najwazniejsze nie trzeba rowno 3) głębokich/nowych ścieżek. Stosuj filtry: FILTR (Ogólne): Pomiń: .(w/b).(shape, device, grad, dtype, req_grad, data, numel), .training, .running_(mean/var), .num_batches_tracked oraz podstawy: .in_features, .in_channels, .out_features, .out_channels, .kernel_size, .p, .groups, .stride, .padding, .dilation. FORMAT WYNIKÓW: SEKCJA: BŁĄD ODCZYTU / BRAK (Zawodzi): Nie istnieją lub zwracają błędny typ (np. Tensor zamiast bool): [wypisz po przecinku same nazwy parametrów z nawiasu, które nie działają jako .xyz] SEKCJA: NOWE / GŁĘBOKIE (Minimum 5 pozycji): nazwa: self.layer.atrybut -> krótki opis po co/kiedy, typ (stan/konfig/statystyka) Uwaga: W sekcji NOWE szukaj głęboko w strukturze modułu (np. metody pomocnicze, flagi wewnętrzne, bufory). Jeśli parametr z nawiasu (np. padding_mode) nie działa bezpośrednio, musi trafić do sekcji BŁĄD.”
.
"""

##########################
"""
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

"""  ##### WARSTWY UCZĄCE SIĘ #####
"""
    self.fc = nn.layer() warstwy juz zostaly opisane w "### OGÓLNE ZASADY ATRYBUTÓW NN ###"
"""

##### warstwa forward #####
"""
    def forward(self, x: Tensor) -> Tensor:
<------- (bez tab)
# 1. KONTROLA WEJŚCIA
print(f"Wejście: {x.shape}") 

# x = self.features(x) / x = self.classifier(x, ... (zalezy od tego czy tam cos bylo)) # przepływ przez całą grupę

# x = self.features[N](x) # przepuszczenie przez N warstwę grupy (indeksowanie)

# x = self.features[1:3](x) # zakres: warstwy od 2. do 3. (slicing)

# feat1 = self.features[0:2](x) 
# feat2 = self.features[2:](feat1)
# x = torch.cat([feat1, feat2], dim=1) # łączenie cech z różnych etapów, ogolna cecha i konkretna jednoczesnie

# skip-connection (Rezdualność)
# po co: kablem przesyłasz info dalej, żeby gradient nie "zgasł" w głębokiej sieci
identity = x
x = self.sub_block(x) # sub_block - grupa warstw nn.Sequential(...)
x = x + identity # mostek: stara informacja + nowa poprawka

# bramkowanie (Gating / Attention)
# model decyduje co ważne: gate=0 olac, gate=1 wziasc
# w Transformerach i sieciach typu SE-Net, żeby odsiać szum z tła
gate = torch.sigmoid(self.gate_layer(x))
x = x * gate # filtrowanie: przechodzi tylko to co istotne
(czemu nie mozna tego poprostu dodac)

global average pooling (Alternatywa dla Flatten)
sprowadza dane 4->2 cechy, wyciąga jedną średnią liczbę z całej mapy cech
x = torch.mean(x, dim=(2, 3)) 
x = x.view(x.size(0), -1) # pozwala na elastyczność i debugowanie wymiarów przed linear

return x

W SKROCIE PRZED TYM NAPISANE:
"xyz"
dalsza czesc:
...

------->
"""


class NN1(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=10,
        hidden_size=64,
        expansion=1,
        act_fn=nn.ReLU,
        dropout_p=0.3,
        use_cat=False,
    ):
        super().__init__()

        self.use_cat = use_cat
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=3,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features=num_classes)
        self.act = act_fn()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.use_cat:
            classifier_in_features = (in_channels + num_classes) * 13 * 13
        else:
            classifier_in_features = num_classes * 13 * 13

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                classifier_in_features,
                hidden_size * expansion,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(
                hidden_size * expansion,
                num_classes,
            ),
        )

    def forward(self, x: Tensor):

        print(f"INPUT: {x.shape}")

        conv_features = self.conv(x)
        conv_features = self.bn(conv_features)
        conv_features = self.act(conv_features)

        pooled = self.pool(conv_features)

        if self.use_cat:
            x_small = F.interpolate(
                x,
                size=(13, 13),
                mode="bilinear",
                align_corners=False,
            )

            pooled = torch.cat(
                [pooled, x_small],
                dim=1,
            )

        print(f"FEATURES: {pooled.shape}")
        out = self.classifier(pooled)
        print(f"OUTPUT: {out.shape}")
        return out


model1 = NN1(
    in_channels=1,
    num_classes=10,
    hidden_size=128,
    expansion=1,
    use_cat=True,
)

dummy_input = torch.randn(1, 1, 28, 28)

output_conv = model1(dummy_input)

print(f"\nKoncowy wynik: {output_conv.shape}")


class MiniTransformerBlock(nn.Module):
    def __init__(self):
        # MINI TRANSFORMER BLOCK
        # CO TO JEST?
        # Transformer = architektura oparta na:
        # - self-attention
        # - LayerNorm
        # - MLP (feed-forward)
        #
        # DLACZEGO SIĘ TEGO UŻYWA?
        #
        # ✔ NLP (LLM, ChatGPT)
        # ✔ Vision (ViT)
        # ✔ multimodal AI
        #
        # CO ROBI TEN BLOK?
        #
        # → uczy relacji między tokenami / elementami
        # → zastępuje klasyczne Conv/RNN
        #
        # WEJŚCIE:
        # (B, T, D)
        #
        # B = batch
        # T = sequence length
        # D = embedding dimension
        #
        # WYJŚCIE:
        # (B, T, D)
        pass

    def forward(self, x):
        pass


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
"""
# 1. NAJNOWSZY STANDARD CONFIGÓW (config.yaml)

model:                      optimizer:
input_size: 784               type: adamw
hidden_size: 512              weight_decay: 1e-4
num_classes: 10             
num_layers: 4                 # Adam-family
dropout_p: 0.2                beta1: 0.9
weight_init: kaiming          beta2: 0.999
                              eps: 1e-8
train:                      
lr: 3e-3                    scheduler:
batch_size: 128               type: cosine
epochs: 5                     warmup_steps: 1000
grad_clip: 1.0                min_lr: 1e-6
label_smoothing: 0.1        
seed: 42                    

system:
  device: cuda
  use_amp: true
  torch_compile: false
```

# 2. WCZYTYWANIE CONFIG
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


# 3. JAK DZIAŁA CONFIG
cfg = {
    "train": {
        "lr": 0.003,
        "batch_size": 128
    }
}

cfg po yaml.safe_load() staje się zwykłym słownikiem Pythona.

# 4. ODCZYTYWANIE HYPERPARAMETRÓW
 
dropout = cfg["model"]["dropout_p"]         optimizer_type = cfg["optimizer"]["type"]
batch_size = cfg["train"]["batch_size"]     lr = cfg["train"]["lr"]   

# 5. GDZIE SĄ HIPERPARAMETRY (NAJWAŻNIEJSZE)

Hiperparametry siedzą w:
- config.yaml -> cfg
- cfg["train"]["lr"] (cfg -> dict)
- trial.suggest_float


# 6. HYDRA
automatyzuje eksperymenty, robi sweeps, robi override parametrów, zapisuje outputy treningów

import hydra

@hydra.main(
    version_base=None,
    config_path=".",
    config_name="config"
)
def main(cfg):

    print(cfg.train.lr)
    print(cfg.model.hidden_size)

if __name__ == "__main__":
    main()

# RÓŻNICA

| YAML               | Hydra               |
| ------------------ | ------------------- |
| cfg["train"]["lr"] | cfg.train.lr        |
| ręczne             | automatyczne        |
| statyczne          | dynamiczne override |
| prostsze           | bardziej SOTA       |

komendy:
Override z terminala:   python train.py train.lr=1e-4
Hydra multirun:         python train.py -m train.lr=1e-3,1e-4


# 7. OBJECTIVE FUNCTION (SERCE AutoML)

objective() siedzi POZA modelem.

Najczęściej:

* w train.py    - gdy wszystko w jednym pliku bo mały projekt
* w sweep.py    - system uruchamiania wielu eksperymentów z różnymi ustawieniami modelu
* w tuning.py   - system automatycznego dostrajania hiperparametrów

# objective() robi:
1. losuje hiperparametry, 2. tworzy model, 3. ustawia optimizer, 4. trenuje model, 5. liczy accuracy / loss, 6. zwraca wynik

# przykład:
"""


def objective(trial: optuna.Trial):

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])

    model = MLP(hidden_size=hidden_size, dropout=dropout).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train(model, optimizer, train_loader, criterion, device)

    val_acc = evaluate(model, val_loader, device)

    return val_acc


"""
# 8. CO ROBI trial.suggest_* ?

# Najważniejsze:

def objective(trial: optuna.trial):
    x1 = trial.suggest_float()
    x2 = trial.suggest_int(...)
    x3 = trial.suggest_categorical(...)
    x4 = trial.suggest_bool(...)
"""

"""
study = optuna.create_study()
study.optimize(objective, n_trials=100)




"""


def objective(trial: optuna.Trial):
    pass
    # params:   name,                 : wyjasnienie
    #           low,                  : min. do wylosowana
    #           high,                 : max. do wylosowania
    #           step=1,               : len miedzy wartosciami
    #           log=False             : sekwencja logarytmiczna
    #           choices=Sequence[...] : wybor z listy

    # params: name, low, high, step, log
    # x = trial.suggest_float(...) : losuje float
    # x = trial.suggest_int(...)   : losuje int

    # params: name, choices
    # x = trial.suggest_categorical(...) : losuje idx, bierze choices[idx]


"""

# 10. SWEEP
Sweep = automatyczne wykonywanie wielu eksperymentów.

Eksperyment 1:
    lr=0.001
    dropout=0.2

Eksperyment 2:
    lr=0.0003
    dropout=0.4

Eksperyment 3:
    lr=0.01
    dropout=0.1

# 11. W&B SWEEPS

```bash
wandb sweep config.yaml
wandb agent PROJECT/SWEEP_ID
```

System:

* uruchamia setki treningów
* zapisuje wyniki
* porównuje modele
* znajduje najlepsze hiperparametry
* pokazuje wykresy
* zapisuje metryki

# 12. RAY TUNE + PBT (bardziej SOTA). (PBT = Population Based Training)

Ray Tune: rozproszony tuning, wiele GPU, wiele maszyn, bardzo szybki AutoML

Podczas treningu słabe modele dostają parametry od najlepszych modeli
czyli hiperparametry mogą zmieniać się W TRAKCIE treningu.

# przykład:
Model A: lr=1e-3
Model B: lr=1e-4 (lepszy)

System:
    kopiuje parametry z B do A

"""
# Realizacja wspoldzielonych modelow:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ray import train, tune
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, hidden_size=256, dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_data():
    x = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64)
    return loader


def trainable(config):

    model = MLP(hidden_size=config["hidden_size"], dropout=config["dropout"])

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    loss_fn = nn.CrossEntropyLoss()

    loader = get_data()

    # TRAIN LOOP
    for epoch in range(3):
        total_loss = 0

        for x, y in loader:
            optimizer.zero_grad()

            preds = model(x)
            loss = loss_fn(preds, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 🔥 WALIDACJA (tutaj uproszczona)
        accuracy = 1.0 / (1.0 + total_loss)  # pseudo accuracy

        # 📡 wysyłasz wynik do Ray Tune
        tune.report(accuracy=accuracy)


results = tune.run(
    trainable,
    config={
        "lr": tune.uniform(1e-5, 1e-2),
        "dropout": tune.uniform(0.1, 0.5),
        "hidden_size": tune.choice([128, 256, 512]),
    },
    num_samples=5,
)

best_trial = results.get_best_trial(metric="accuracy", mode="max", scope="last")

print("\n=== BEST TRIAL ===")
assert best_trial is not None

print(
    "accuracy    :", best_trial.last_result["accuracy"]
)  # "last_result" is not a known attribute of "None"
print("lr          :", best_trial.config["lr"])
print("dropout     :", best_trial.config["dropout"])
print("hidden_size :", best_trial.config["hidden_size"])

for i, trial in enumerate(results.trials):
    print(f"\nMODEL {i + 1}")

    print("accuracy    :", trial.last_result["accuracy"])
    print("lr          :", trial.config["lr"])
    print("dropout     :", trial.config["dropout"])
    print("hidden_size :", trial.config["hidden_size"])
"""

# 18. PRAWDZIWY FLOW W SOTA AI

1. config.yaml definiuje default model
2. Hydra ładuje config
3. Optuna / Ray Tune generują nowe parametry
4. objective() tworzy model
5. training loop trenuje model
6. accuracy wraca do AutoML systemu
7. system wybiera najlepszy eksperyment

"""

### Load Data ###

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

train_dataset = datasets.MNIST(  # rozpisac mozliwosci tego  i jak to dziala i co robi
    root="/dataset", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(
    root="/dataset", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

""" WYJASNIENIA: datasets.MNIST, DataLoader() 
class MNIST( : 
    root,                   : 
    train=True,             : 
    transform=None,         : 
    target_transform=None,  : 
    download=False          : 
)

class DataLoader( :
    dataset,            : 
    batch_size=1,       : 
    shuffle=None,       : 
    sampler=None,       : 
    batch_sampler=None, : 
    num_workers=0,      : 
    collate_fn=None,    : 
    pin_memory=False,   : 
    drop_last=False,    : 
    timeout=0,          : 
    worker_init_fn=None,          : 
    multiprocessing_context=None, : 
    generator=None,               : 
    prefetch_factor=None,         : 
    persistent_workers=False,     : 
    pin_memory_device="",         : 
    in_order=True,                : 
)
"""


### Initalize network ###
class NN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=64, expansion=1):
        super().__init__()

        self.fc1 = nn.Linear(
            in_features=input_size,
            out_features=hidden_size * expansion,
        )

        self.fc2 = nn.Linear(
            in_features=hidden_size * expansion,
            out_features=num_classes,
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NN(input_size=input_size, num_classes=num_classes).to(device)

### Loss and optimizer ###
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

""" WYJASNIENIA: nn.CrossEntropyLoss(), optim.Adam()
class CrossEntropyLoss( : 
    weight=None,        : 
    size_average=None,  : 
    ignore_index=-100,  : 
    reduce=None,        : 
    reduction="mean",   : 
    label_smoothing=0   : 
)

class Adam( : 
    params,               : 
    lr=1e-3,              : 
    betas=(0.9, 0.999),   : 
    eps=1e-8,             : 
    weight_decay=0,       : 
    amsgrad=False,        : 
    foreach=None,         : 
    maximize=False,       : 
    capturable=False,     : 
    differentiable=False, : 
    fused=None,           : 
    decoupled_weight_decay=False, : 
)
"""

### Trainer Network ###
data: torch.Tensor
targets: torch.Tensor

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(
            device
        )  # data i target musi byc na tym samym urzadzeniu co model przed "scores = model(data)"
        targets = targets.to(device)

        data = torch.flatten(
            data, start_dim=1
        )  # data nalezy dostosowac do wejscia w model(data), [64, 1, 28, 28] -> [64, 784]
        # [64, 1, 28, 28] - 64 obrazki 28x28, [64, 784] 64 linie cyfr dlugosci 784
        ## forward
        scores = model(
            data
        )  # zwraca T - [batch_size, num_classes], tzw. LOGITS (surowe przyszle przewidywania modelu)
        loss: torch.Tensor = criterion(
            scores, targets
        )  # zwraca int (skalar - przewidywania), mierzy różnice prediction (scores) od prawdy (targets)
        # mniejszy loss = lepszy model (sygnał że model idzie w dobrą stronę)

        ## backward
        # gradient (.grad) - int, jaka zmiana wyniku (wynik - loss), gradient > 0 (zmniejsz wagę), gradient < 0 (zwiększ wagę)

        optimizer.zero_grad()  # zeruje gradienty w wagach modelu, zawsze przed .backward()
        # PyTorch domyślnie sumuje gradienty, bez model uczy się na stary + nowy gradient

        loss.backward()  # liczy gradienty i loss

        ## gradient descent or adam step
        optimizer.step()  # używa wcześniej policzonych gradientów (.grad) i aktualizuje wagi modelu

""" WYJASNIENIA for epoch in range(num_epochs):
Jak przebiega pętla:
1) DataLoader zwraca kolejny batch danych
2) batch trafia na device (CPU/GPU)
3) przygotowujesz dane (reshape, augmentacje itd.)
4) model(data) -> forward pass
5) model zwraca scores/logits
6) criterion(scores, targets) liczy loss
7) optimizer.zero_grad() czyści stare gradienty
8) loss.backward() liczy gradienty dla wszystkich wag
9) optimizer.step() aktualizuje wagi
10) następny batch
11) po przejściu wszystkich batchy kończy się epoch
12) zaczyna się kolejny epoch
13) po ostatnim epochu trening się kończy

# jakie sa najnowsze standardy:
PyTorch Lightning
HuggingFace Trainer
Accelerate
DeepSpeed

"""


### Check accuracy on training & test ###
def check_accuracy(loader: DataLoader, model: nn.Module) -> float:
    dataset = cast(MNIST, loader.dataset)

    if dataset.train:
        print("Checking accuracy on training data.")
    else:
        print("Checking accuracy on test data.")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            inputs = torch.flatten(inputs, start_dim=1)

            scores: Tensor = model(inputs)

            _, predictions = scores.max(
                1
            )  # .max(dim=None "po ktory wymiarze liczyc max()", keepdim=False "") -> Tensor or (Tensor, Tensor)

            num_correct += (
                (predictions == targets).sum().item()
            )  # liczy ile wynikow jest zgodnych z odpowiedzia

            num_samples += predictions.size(
                0
            )  # opcjonalnie: num_samples += len(predictions)

        accuracy = num_correct / num_samples * 100

        print(
            f"Accuracy: {accuracy:.2f}%"  # poprawne/wszystkie * 100 bo dokladnosc w procentach
        )

    model.train()  # po model.eval() - tryb testowania, wlacza sie zwykle model.train() - tryb uczenia sie

    return accuracy


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
