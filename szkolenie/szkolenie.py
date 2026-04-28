import logging
import os
import sys
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class Szkolenie:
    def __init__(self, tasks: Optional[Union[list, str]] = None):
        logger.debug("Starting program tests...")

        if not tasks:
            return

        tasks = tasks if isinstance(tasks, list) else [tasks]

        for func in tasks:
            name = func.__name__
            logger.info(f"Running: {name}")
            try:
                func()
                logger.info(f"ACTION: {name} was successful!")
            except Exception as e:
                line = sys.exc_info()[-1].tb_next.tb_lineno
                logger.error(f"FAILED: {name} at line {line} with error '{e}'")


def task_numpy():
    print(f"version: {np.version.full_version}")  # 2

    print(np.zeros(10))  # 3
    A = np.array([1, 2, 3], dtype=np.uint8)
    print(A.nbytes)  # 4

    print(np.info(np.add))  # 5

    A = np.zeros(10)[4] = 1  # 6
    print(A)
    print(np.arange(10, 50))  # 7

    print(np.arange(1, 4) * -1)  # 8

    print(np.reshape(np.arange(9), [3, -1]))  # 9

    B = np.array([1, 2, 0, 0, 4, 0])
    print(B[B > 0])  # 10


def task_():
    data = {
        "OrderID": [1001, 1002, 1003, 1004, 1005, 1006],
        "Customer": ["Alice", "Bob", "Alice", "Derek", "Eve", "Bob"],
        "Product": ["Laptop", "Monitor", "Mouse", "Laptop", "Mouse", "Monitor"],
        "Quantity": [1, 2, 3, 1, 2, 1],
        "Price": [1200, 300, 25, 1200, 25, 300],
        "Date": pd.to_datetime(
            [
                "2024-05-01",
                "2024-05-03",
                "2024-05-04",
                "2024-05-05",
                "2024-05-06",
                "2024-05-06",
            ]
        ),
    }

    df = pd.DataFrame(data)  # #tworzenie_df
    df.head(5)  # #podglad_danych
    df.describe()  # #statystyki_opisowe

    # nowa_kolumna
    df["Total"] = df["Quantity"] * df["Price"]

    # grup #suma_na_klienta
    df2 = df.groupby("Customer")["Total"].sum()
    print(f"{df2}")

    # srednia_wartosc_zamowienia
    df3 = df.groupby("Customer")["Total"].mean()
    print(f"{df3}")

    # top_sprzedaz #szukanie_max
    df4 = df.groupby("Product")["Quantity"].sum().idxmax()
    print(f"Najlepszy produkt: {df4}")

    # filtrowanie_po_dacie #maska_logiczna
    df5 = df[df["Date"] > "2024-05-03"]
    print(df5)

    # #sortowanie_malejaco #ranking_total
    df6 = df.sort_values(by="Total", ascending=False)
    print(df6)

    # tabela_przestawna
    pivot = df.pivot_table(
        index="Date", columns="Product", values="Total", aggfunc="sum", fill_value=0
    )
    print(pivot)


def _pandas():
    base_path = os.path.dirname(__file__)
    csv_path = os.path.join(base_path, "plik.csv")

    # Wczytujemy dane (zakładamy, że plik ma kolumny: name, age, city, salary)
    df = pd.read_csv(csv_path)

    # 1. TWORZENIE I STRUKTURA
    print("--- INFO ---")
    df.info()  # Informacje techniczne
    print("\n--- STATYSTYKI ---")
    print(df.describe())  # Statystyki matematyczne
    print("\n--- KOLUMNY ---")
    print(df.columns)  # Lista nazw kolumn

    # 2. DOSTĘP I MODYFIKACJA
    print("\n--- DOSTĘP DO DANYCH ---")
    # Wybór konkretnej komórki: kolumna "name", wiersz 0
    print(f"Pierwsze imię: {df['name'][0]}")

    # Zmiana wartości: wiersz o indeksie 4, kolumna "name" na "Ewa"
    df.loc[4, "name"] = "Ewa"

    # Maska logiczna: zwraca Series z True/False
    starszy_niz_27 = df["age"] > 27
    print("Czy osoby mają > 27 lat?\n", starszy_niz_27)

    # 3. TRANSFORMACJA DANYCH
    # Nowa kolumna na podstawie obliczeń (10% pensji)
    df["bonus"] = df["salary"] * 0.1

    # Użycie lambda: przypisanie poziomu na podstawie wieku
    df["level"] = df["age"].apply(lambda x: "Junior" if x < 30 else "Senior")

    # Sortowanie tabeli według wieku malejąco
    df_sorted = df.sort_values(by="age", ascending=False)

    # 4. CZYSZCZENIE I ŁĄCZENIE
    # Usuwanie wierszy z brakami (NaN)
    df_clean = df.dropna()

    # Zastępowanie braków w konkretnej kolumnie (np. jeśli miasto jest puste)
    df["city"] = df["city"].fillna("Inny")

    # Łączenie tabel (Merge) - tworzymy mały słownik pomocniczy
    dane_dodatkowe = pd.DataFrame(
        {"name": ["Ewa", "Marek"], "department": ["HR", "IT"]}
    )
    # Łączymy naszą główną tabelę z nową po kolumnie "name"
    df = pd.merge(df, dane_dodatkowe, on="name", how="left")

    # 5. AGREGACJA (ANALIZA)
    print("\n--- ŚREDNIA PENSJA PO MIASTACH ---")
    print(df.groupby("city")["salary"].mean())

    print("\n--- TABELA PRZESTAWNA ---")
    # Średnie zarobki w zależności od miasta i poziomu (Junior/Senior)
    pivot = df.pivot_table(
        values="salary", index="city", columns="level", aggfunc="mean"
    )
    print(pivot)

    # Wyświetlenie końcowego efektu
    print("\n--- FINALNY HEAD ---")
    print(df.head())


def _seaborn_():
    df = sns.load_dataset("titanic")
    sns.violinplot(data=df, x="class", y="age", hue="alive")
    plt.show()


if __name__ == "__main__":
    Szkolenie(_seaborn_)
