import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def visualize_classifier(classifier, X, y, title=""):
    # Określenie granic wykresu z marginesem, aby punkty nie dotykały krawędzi
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Rozdzielczość tła (im mniejsza wartość, tym dokładniejszy rysunek granic decyzji)
    mesh_step_size = 0.01

    # Tworzenie gęstej siatki punktów (tła), którą klasyfikator będzie musiał "pokolorować"
    x_vals, y_vals = np.meshgrid(
        np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size)
    )

    output = classifier.predict(
        np.c_[x_vals.ravel(), y_vals.ravel()]
    )  # Przewidywanie klasy dla każdego punktu tła

    output = output.reshape(x_vals.shape)  # Dopasowanie wyników do kształtu siatki tła

    plt.figure()  # rys. wykres
    plt.title(title)

    # Rysowanie kolorowych obszarów decyzji (mapa tła)
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    # Nałożenie rzeczywistych punktów danych (X) na pokolorowane tło
    plt.scatter(
        X[:, 0], X[:, 1], c=y, s=75, edgecolors="black", linewidth=1, cmap=plt.cm.Paired
    )

    # Ustawienie zakresów osi i czytelnych jednostek (co 1.0)
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))

    plt.show()


base_path = os.path.dirname(__file__)
input_file = os.path.join(base_path, "data_decision_trees.txt")

data = np.loadtxt(input_file, delimiter=",")

outlier = np.array([[5, 5, 1.0]])
data = np.vstack([data, outlier])

X, y = data[:, :-1], data[:, -1]

class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])

plt.figure()
plt.scatter(
    class_0[:, 0],
    class_0[:, 1],
    s=75,
    facecolors="black",
    edgecolors="black",
    linewidth=1,
    marker="x",
)
plt.scatter(
    class_1[:, 0],
    class_1[:, 1],
    s=75,
    facecolors="white",
    edgecolors="black",
    linewidth=1,
    marker="o",
)
plt.title("Input data")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

params = {"random_state": 0, "max_depth": 8}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, "Training dataset")

y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, "Test dataset")

class_names = ["Class-0", "Class-1"]
print("\n" + "#" * 40)
print("\nClassifier performance on training dataset\n")
print(
    classification_report(
        y_train, classifier.predict(X_train), target_names=class_names
    )
)
print("#" * 40 + "\n")

print("#" * 40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#" * 40 + "\n")

plt.show()
