import os
import shutil

import kagglehub
import pandas as pd

path = kagglehub.dataset_download(
    "ibrahimshahrukh/tesla-stock-price-historical-dataset-2010-2025"
)

try:
    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(path, csv_file))

    print(df.head())
    print("Praca na danych zakończona.")

finally:
    print(f"Sprzątanie... Usuwam folder: {path}")
    shutil.rmtree(path)
