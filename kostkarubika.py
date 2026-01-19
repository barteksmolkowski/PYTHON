# przód kostki czerwony, prawa niebieski, lewy zielony, dół żółty numery liczą się warstwami
# przedniej do tylnej a ścianami od lewej do prawej.
# 1 -> biały-zielony-czerwony
# 3 -> biały-niebieski-czerwony
# 9 -> biały-niebieski-pomarańczowy
# 12 -> niebieski-pomarańczowy
# ostatni -> dół tył prawy żółty-niebieski-pomarańczowy

import copy

tablica = {"boki": {}, "rogi": {}}
tablica["boki"] = {
    1: [2, ["white", "red"]],
    2: [4, ["white", "green"]],
    3: [5, ["white", "blue"]],
    4: [7, ["white", "orange"]],
    5: [9, ["red", "green"]],
    6: [10, ["red", "blue"]],
    7: [11, ["orange", "green"]],
    8: [12, ["orange", "blue"]],
    9: [14, ["yellow", "red"]],
    10: [16, ["yellow", "green"]],
    11: [17, ["yellow", "blue"]],
    12: [19, ["yellow", "orange"]],
}

tablica["rogi"] = {
    1: [1, ["white", "red", "green"]],
    2: [3, ["white", "red", "blue"]],
    3: [6, ["white", "orange", "green"]],
    4: [8, ["white", "orange", "blue"]],
    5: [13, ["yellow", "red", "green"]],
    6: [15, ["yellow", "red", "blue"]],
    7: [18, ["yellow", "orange", "green"]],
    8: [20, ["yellow", "orange", "blue"]],
}

pamiectablica = copy.deepcopy(tablica)


def Krzyz(tablica):
    # 0=white, 1=red, 2=orange, 3=green, 4=blue, 5=yellow
    wyniki = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}
    boki, pamiecboki = tablica["boki"], pamiectablica["boki"]

    numeryBok = {
        0: [2, 4, 5, 7],
        1: [2, 9, 10, 14],
        2: [7, 11, 12, 19],
        3: [4, 9, 11, 16],
        4: [5, 10, 12, 17],
        5: [14, 16, 17, 19],
    }

    for i, numery in numeryBok.items():
        print(f"Ściana {i}: numery {numery}")

        for wartosc in numery:
            print(f"  sprawdzam numer {wartosc}")
            print(f"  liczba elementów w boki: {len(tablica['boki'])}")

    return wyniki


Krzyz(tablica)


# def ruch(numer):
#     if numer == 0:
#         0

