import math
import sys
from typing import List

def oblicz(linie: List[str]) -> int:
    max_dlugosc = max(len(l) for l in linie)
    siatka = [list(l.ljust(max_dlugosc)) for l in linie]

    wiersze = len(siatka)
    cols = max_dlugosc

    odzielacz = [all(siatka[r][c] == ' ' for r in range(wiersze)) for c in range(cols)]

    grupy = []
    c = 0
    while c < cols:
        if not odzielacz[c]:
            start = c
            while c < cols and not odzielacz[c]:
                c += 1
            grupy.append((start, c - 1))
        else:
            c += 1

    wynik = 0
    for start, koniec in grupy:
        operator = siatka[-1][start]

        liczby = []
        for kol in range(koniec, start - 1, -1):
            znaki = ''.join(siatka[r][kol] for r in range(wiersze - 1))
            num = znaki.replace(' ', '')
            if num == '':
                num = '0'
            liczby.append(num)

        wynik_liczby = [int(x) for x in liczby]
        match operator:
            case "+":
                wartosc = sum(wynik_liczby)
            case "*":
                wartosc = math.prod(wynik_liczby)

        wynik += wartosc

    return wynik

wejscie = []
for linia in sys.stdin:
    linia = linia.rstrip('\n')
    if linia == '':
        break
    wejscie.append(linia)
if wejscie:
    print(oblicz(wejscie))
