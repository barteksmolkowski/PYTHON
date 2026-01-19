import sys

lista = []
for linia in sys.stdin:
    linia = linia.strip()
    if not linia:
        break
    lista.append(linia)

n_lista = []
for wiersz in lista:
    n_wiersz = []
    for el in wiersz:
        n_wiersz.append([el])
    n_lista.append(n_wiersz)

lista = n_lista

def pokolenie(poprzednie, aktualne, kierunek):
    nazwy_kierunek = {0: "L", 1: "P"}
    liczba_rozdzielen = 0
    for i, el in enumerate(poprzednie):
        if el[0] == "S":
            aktualne[i] = ["|"]

    for i, el in enumerate(poprzednie):
        if el[0] == "|":
            if aktualne[i][0] == "^":
                try:
                    if kierunek == "L":
                        aktualne[i - 1] = ["|"]
                    else:
                        aktualne[i + 1] = ["|"]
                    liczba_rozdzielen += 1
                except IndexError:
                    pass
            else:
                aktualne[i] = ["|"]

    return aktualne, liczba_rozdzielen

def zlec_mozliwoscia(lista_ruchow): # binarnie w lewo albo prawo i tak liczy
    for i in range(1, len(lista)):
        print(f"zlec_mozliwoscia: {lista[i - 1], lista[i], lista_ruchow[i]}")
        el, liczba_rozdzielen = pokolenie(lista[i - 1], lista[i], lista_ruchow[i])
        n_lista.append(el)
        suma += liczba_rozdzielen

def wygeneruj_mozliwosci(liczba):
    lista = []

    for i in range(2 ** liczba):
        list_mozliwosci = list(bin(i)[2:].zfill(liczba))
        mozliwosc = []

        for el in list_mozliwosci:
            mozliwosc.append("L" if el == "0" else "P")
        lista.append(mozliwosc)

    return lista

def sprawdz_tab(tablica):
    0

n_lista = []
n_lista.append(lista[0])
suma = 0
ilosc = 0
for i in range(len(lista)):
    if ["^"] in lista[i]:
        ilosc += 1
        
ilosc_BIN_rozwidlen = wygeneruj_mozliwosci(ilosc)
print(ilosc_BIN_rozwidlen)


lista = n_lista
for wiersz in lista:
    print("".join(znak[0] for znak in wiersz))

print(f"wynik: {suma}")