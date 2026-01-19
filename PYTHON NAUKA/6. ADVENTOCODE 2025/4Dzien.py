import sys

lista = []
for linia in sys.stdin:
    linia = linia.strip()
    if not linia:
        break
    lista.append(linia)

def usun_pola(lista, do_usuniecia):
    for (y, x) in do_usuniecia:
        lista[y][x] = "."
    return lista

def policz_sasiadow(x, y, lista):
    ruchy = [(-1,-1), (0,-1), (1,-1),
             (-1, 0),         (1, 0),
             (-1, 1), (0, 1), (1, 1)]
    wysokosc = len(lista)
    szerokosc = len(lista[0])
    ilosc = 0
    for dx, dy in ruchy:
        nx, ny = x + dx, y + dy
        if 0 <= ny < wysokosc and 0 <= nx < szerokosc:
            if lista[ny][nx] == "@":
                ilosc += 1
    return ilosc

def usun_dostepnych_sasiadow(lista):
    lista = [list(wiersz) for wiersz in lista]
    total_removed = 0
    while True:
        do_usuniecia = []
        wysokosc = len(lista)
        szerokosc = len(lista[0])

        for y in range(wysokosc):
            for x in range(szerokosc):
                if lista[y][x] == "@":
                    if policz_sasiadow(x, y, lista) < 4:
                        do_usuniecia.append((y, x))

        if not do_usuniecia:
            break

        total_removed += len(do_usuniecia)
        lista = usun_pola(lista, do_usuniecia)

    return total_removed

print(usun_dostepnych_sasiadow(lista))

