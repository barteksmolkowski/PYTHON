class DataSubset:
    def __init__(self, data_dict):
        self.data = data_dict

    def data_preprocessing(self, MinMaxNormalization=False):
        nowa_data = {}

        for nazwa_cechy, wartosci in self.data.items():
            unikalne = list(dict.fromkeys(wartosci))
            dlugosc = len(unikalne)
            typ = type(wartosci[0])

            if dlugosc == 1:
                continue

            elif dlugosc == 2:
                if typ == str:
                    for kategoria in unikalne:
                        nowa_data[f"{nazwa_cechy}_{kategoria}"] = [
                            1 if v == kategoria else 0 for v in wartosci
                        ]
                elif typ in (int, float):
                    nowa_data[nazwa_cechy] = wartosci
                else:
                    continue

            else:
                if typ == str:
                    for kategoria in unikalne:
                        nowa_data[f"{nazwa_cechy}_{kategoria}"] = [
                            1 if v == kategoria else 0 for v in wartosci
                        ]
                elif typ in (int, float):
                    if MinMaxNormalization:
                        min_wart = min(wartosci)
                        max_wart = max(wartosci)
                        if max_wart - min_wart == 0:
                            n_wartosci = [0 for _ in wartosci]
                        else:
                            n_wartosci = [
                                (el - min_wart) / (max_wart - min_wart)
                                for el in wartosci
                            ]
                        nowa_data[nazwa_cechy] = n_wartosci
                    else:
                        srednia = sum(wartosci) / len(wartosci)
                        if srednia == 0:
                            n_wartosci = wartosci
                        else:
                            n_wartosci = [el / srednia for el in wartosci]
                        nowa_data[nazwa_cechy] = n_wartosci
                else:
                    continue

        self.data = nowa_data
        return self

    def __repr__(self):
        return str(self.data)


class DataLoader:
    def __init__(self):
        self.data = []

    def load_data(self, data):
        self.data = data

    def _to_column_dict(self, keys=None, ignoruj=None):
        if ignoruj is None:
            ignoruj = []
        elif isinstance(ignoruj, str):
            ignoruj = [ignoruj]

        keys = (
            [k for k in self.data[0].keys() if k not in ignoruj]
            if keys is None
            else keys
        )

        kolumny = {k: [] for k in keys}
        for wiersz in self.data:
            for k in keys:
                kolumny[k].append(wiersz[k])
        return kolumny

    def pobierz_element(
        self,
        czyWszystkie=True,
        ignoruj=None,
        lista=None,
        dataPreprocessing=False,
        MinMaxNormalization=False,
    ):
        return DataSubset(
            self._to_column_dict(keys=lista, ignoruj=ignoruj if ignoruj else [])
        )


data = [
    {
        "id": 1,
        "size": 42,
        "color": "red",
        "gender": "M",
        "price": 99.99,
        "weight": 0.5,
        "bought": 1,
    },
    {
        "id": 2,
        "size": 38,
        "color": "blue",
        "gender": "F",
        "price": 79.50,
        "weight": 0.45,
        "bought": 0,
    },
    {
        "id": 3,
        "size": 40,
        "color": "green",
        "gender": "M",
        "price": 89.00,
        "weight": 0.48,
        "bought": 1,
    },
    {
        "id": 4,
        "size": 44,
        "color": "black",
        "gender": "F",
        "price": 120.00,
        "weight": 0.55,
        "bought": 1,
    },
    {
        "id": 5,
        "size": 39,
        "color": "red",
        "gender": "F",
        "price": 75.00,
        "weight": 0.47,
        "bought": 0,
    },
]

dataloader = DataLoader()
dataloader.load_data(data)

elementy = dataloader.pobierz_element(czyWszystkie=True, ignoruj="id")
elementy.data_preprocessing(MinMaxNormalization=True)
print(elementy)
