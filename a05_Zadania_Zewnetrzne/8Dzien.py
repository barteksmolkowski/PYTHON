import sys

nlista = []
for linia in sys.stdin:
    linia = linia.strip()
    if not linia:
        break
    nlista.append(linia)


def podaj_najblizszy(punkt: tuple, wszystkie: list[tuple]):
    inf = float("inf")
    min_dystans = inf
    najblizszy = None

    x1, y1, z1 = punkt

    for x2, y2, z2 in wszystkie:
        if (x1, y1, z1) == (x2, y2, z2):
            continue

        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        dystans = dx * dx + dy * dy + dz * dz

        if dystans < min_dystans:
            min_dystans = dystans
            najblizszy = (x2, y2, z2)

    return najblizszy, min_dystans


nlista = [
    "162,817,812",
    "57,618,57",
    "906,360,560",
    "592,479,940",
    "352,342,300",
    "466,668,158",
    "542,29,236",
    "431,825,988",
    "739,650,466",
    "52,470,668",
    "216,146,977",
    "819,987,18",
    "117,168,530",
    "805,96,715",
    "346,949,466",
    "970,615,88",
    "941,993,340",
    "862,61,35",
    "984,92,344",
    "425,690,689",
]
lista = []

for el in nlista:
    str_l = el.split(",")
    print(len(str_l))
    lista.append((int(str_l[0]), int(str_l[1]), int(str_l[2])))

print(lista)
