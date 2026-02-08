import pandas as pd

data = {
    "name": [
        "Marek",
        "Ania",
        "Piotr",
        "Kasia",
        "Tomek",
        "Ola",
        "Jurek",
        "Basia",
        "Andrzej",
        "Magda",
    ],
    "age": [42, 28, 35, 22, 50, 31, 45, 33, 29, 37],
    "city": [
        "Warszawa",
        "Wrocław",
        "Warszawa",
        "Łódź",
        "Wrocław",
        "Łódź",
        "Warszawa",
        "Łódź",
        "Wrocław",
        "Warszawa",
    ],
    "salary": [5400, 7200, 6100, 4800, 9500, 5900, 8300, 6200, 5100, 7000],
}

df = pd.DataFrame(data)

print("--- INFO O RAMCE DANYCH ---")
df.info()

print("\n--- STATYSTYKI OPISOWE ---")
print(df.describe())

print("\n--- PIERWSZE IMIĘ Z LISTY ---")
print(df["name"][0])

print("\n--- CZY OSOBA MA WIĘCEJ NIŻ 27 LAT? (Maska logiczna) ---")
print(df["age"] > 27)

df["bonus"] = df["salary"] * 0.1

df["level"] = df["age"].apply(lambda x: "Junior" if x < 30 else "Senior")

# CZYSZCZENIE DANYCH
df2 = df.dropna()
df3 = df.fillna(0)

# ŁĄCZENIE TABEL
dane_dzialy = {
    "name": ["Marek", "Ania", "Piotr", "Kasia", "Tomek"],
    "department": ["IT", "IT", "Marketing", "HR", "Sales"],
}
nowy_df = pd.DataFrame(dane_dzialy)

# Merge przed zmianą imienia, żeby dopasować działy
df_final = pd.merge(df, nowy_df, on="name", how="left")

# FINALNE SZLIFOWANIE (Zmiana imienia i uzupełnienie braków)
# Teraz zmieniamy Tomka (indeks 4) na Ewę
df_final.loc[4, "name"] = "Ewa"
# Wypełniamy działy dla osób spoza listy (Jurek, Magda itd.)
df_final["department"] = df_final["department"].fillna("Inny")

# WYŚWIETLANIE WYNIKÓW (Wszystkie zestawienia z obu kodów)
print("\n--- DANE POSORTOWANE WEDŁUG WIEKU ---")
print(df_final.sort_values(by="age"))

print("\n--- ŚREDNIA PENSJA W KAŻDYM MIEŚCIE ---")
print(df_final.groupby("city")["salary"].mean())

print("\n--- POŁĄCZONA TABELA Z POZIOMEM I DZIAŁEM ---")
print(df_final[["name", "age", "city", "salary", "department", "level"]])

print("\n--- CAŁA TABELA (FINALNA) ---")
print(df_final)

# Tworzenie tabeli przestawnej:
# Średnie zarobki w podziale na miasto (wiersze) i poziom (kolumny)
pivot = df_final.pivot_table(
    values="salary",
    index="city",
    columns="level",
    aggfunc="mean",
)

print("\n--- TABELA PRZESTAWNA (ŚREDNIE ZAROBKI) ---")
print(pivot)
