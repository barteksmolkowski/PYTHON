import os

nazwa = "folder/dane.txt"
folder = os.path.dirname(nazwa)

os.makedirs(folder, exist_ok=True)

try:
    with open(nazwa, "w+") as plik:
        print(f"Otwarto plik: {plik}")
        plik.write("siema")
except FileNotFoundError:
    print("Błąd")
