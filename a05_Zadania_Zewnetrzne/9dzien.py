# import sys

# nlista = []
# for linia in sys.stdin:
#     linia = linia.strip()
#     if not linia:
#         break
#     nlista.append(linia)


def stworz_tab(lista):
    szer_max, wys_max = 0, 0

    for x, y in lista:
        szer_max = max(szer_max, x)
        wys_max = max(wys_max, y)

    tablica = [["." for _ in range(szer_max)] for _ in range(wys_max)]

    for x, y in lista:
        tablica[y - 1][x - 1] = "#"

    return tablica


def zaznacz_punkty(rogi, wszystkie):
    print("zaznacza rogi na wszystkie")
    max_x = max(x for x, _ in zip(rogi, wszystkie))
    max_y = max(y for _, y in zip(rogi, wszystkie))

    tab = [["." for _ in range(max_x)] for _ in range(max_y)]

    for x, y in wszystkie:
        tab[y - 1][x - 1] = "#"
    for x, y in rogi:
        tab[y - 1][x - 1] = "O"

    return tab


def znajdz_rogi(lista):
    inf = float("inf")
    p = {
        "s": [(inf, None), []],
        "w": [(-inf, None), []],
        "d": [(None, inf), []],
        "a": [(None, -inf), []],
    }

    for x, y in lista:
        if x < p["s"][0][0]:
            p["s"][0] = (x, y)
            p["s"][1] = [(x, y)]

        elif x == p["s"][0][0]:
            p["s"][1].append((x, y))

        if x > p["w"][0][0]:
            p["w"][0] = (x, y)
            p["w"][1] = [(x, y)]
        elif x == p["w"][0][0]:
            p["w"][1].append((x, y))

        if y < p["d"][0][1]:
            p["d"][0] = (x, y)
            p["d"][1] = [(x, y)]
        elif y == p["d"][0][1]:
            p["d"][1].append((x, y))

        if y > p["a"][0][1]:
            p["a"][0] = (x, y)
            p["a"][1] = [(x, y)]
        elif y == p["a"][0][1]:
            p["a"][1].append((x, y))

        znalezione = p["s"][1] + p["w"][1] + p["d"][1] + p["a"][1]

    return znalezione


nlista = ["7,1", "11,1", "11,7", "9,7", "9,5", "2,5", "2,3", "7,3"]
lista = []

for el in nlista:
    l1, l2 = el.split(",")
    lista.append((int(l1), int(l2)))

tablica = stworz_tab(lista)

rogi = znajdz_rogi(lista)
mapa = zaznacz_punkty(rogi, lista)

for i in mapa:
    print(i)
