import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# 1. DANE
X = np.array(
    [
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
        [4, 4],
        [4, 0],
        [3, 1],
        [2, 3],
        [3, 3],
        [5, 5],
        [2, 3],
        [4, 0],
    ]
)
X_extra = np.random.uniform(0, 6, size=(50, 2))
X = np.vstack([X, X_extra])


# 2. MODEL I TRENOWANIE
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Pobieramy wyniki
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 3. WIZUALIZACJA (Matplotlib)
plt.figure(figsize=(8, 6))

# Rysujemy punkty danych (kolorujemy je według etykiet z kmeans)
# c=labels nadaje kolory grupom, cmap='viridis' to ładna paleta barw
plt.scatter(X[:, 0], X[:, 1], c=labels, s=100, cmap="viridis", label="Punkty danych")

# Rysujemy środki klastrów (Centroidy)
# marker='X' - kształt krzyżyka, s=300 - duży rozmiar, c='red' - czerwony kolor
plt.scatter(
    centers[:, 0], centers[:, 1], c="red", s=300, marker="X", label="Środki (Centroidy)"
)

# Dodajemy nowy punkt z Twojego przykładu [0,0] i sprawdzamy go na wykresie
new_point = np.array([[0, 0]])
pred = kmeans.predict(new_point)
plt.scatter(
    new_point[:, 0],
    new_point[:, 1],
    c="black",
    s=150,
    marker="o",
    label=f"Nowy punkt [0,0] -> Grupa {pred[0]}",
)

# Estetyka wykresu
plt.title("Wizualizacja K-Means (3 grupy)")
plt.xlabel("Oś X")
plt.ylabel("Oś Y")
plt.legend()  # Wyświetla legendę z opisami
plt.grid(True, linestyle="--", alpha=0.6)  # Dodaje pomocniczą siatkę

# KLUCZOWA KOMENDA - bez niej okno się nie otworzy
plt.show()
