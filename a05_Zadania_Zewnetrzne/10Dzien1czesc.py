def stworz_liste(lista):
    return [
        (
            list(linia.split()[0][1:-1]),
            [list(map(int, el[1:-1].split(","))) for el in linia.split()[1:-1]]
        )
        for linia in lista
    ]

def przelacz_kontrolki(lista_przelaczen):
    licznik = Counter()
    for przycisk in lista_przelaczen:
        licznik.update(przycisk)
    return [pozycja for pozycja, ilosc in licznik.items() if ilosc % 2 == 1]

def stworz_poczatek(lista_kontrolek):
    return ["." for _ in lista_kontrolek]

def zastosuj_przelaczenia(stan, przełączone_pozycje):
    nowy_stan = stan.copy()
    for p in przełączone_pozycje:
        if nowy_stan[p] == '.':
            nowy_stan[p] = '#'
        else:
            nowy_stan[p] = '.'
    return nowy_stan

def sprawdz_przelaczenie(docelowy, aktualny):
    return docelowy == aktualny

def stworz_mozliwosci(lista, ilosc):
    return list(combinations(lista, ilosc))

def policz_min_przelaczniki(linia):
    docelowy, przyciski = linia
    stan = stworz_poczatek(docelowy)

    ilosc = 1
    while True:
        mozliwosci = stworz_mozliwosci(przyciski, ilosc)
        for mozliwosc in mozliwosci:
            przełączone = przelacz_kontrolki(mozliwosc)
            nowy_stan = zastosuj_przelaczenia(stan, przełączone)
            if sprawdz_przelaczenie(docelowy, nowy_stan):
                return ilosc
        ilosc += 1

import sys
from itertools import combinations
from collections import Counter

lista = []
for linia in sys.stdin:
    linia = linia.strip()
    if not linia:
        break
    lista.append(linia)

print(sum([policz_min_przelaczniki(l) for l in stworz_liste(lista)]))