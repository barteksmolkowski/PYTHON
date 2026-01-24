import sys

lista = []
for linia in sys.stdin:
    linia = linia.strip()
    if not linia:
        break
    lista.append(linia)


def znajdz_max_liczbe(ciag_cyfr, dlugosc):
    wynik = ""
    pozycja = 0
    while len(wynik) < dlugosc:
        ile_pozostalo = dlugosc - len(wynik)
        maksymalny_indeks = len(ciag_cyfr) - ile_pozostalo

        najlepsza_cyfra = "0"
        indeks_najlepszej = -1
        for i in range(pozycja, maksymalny_indeks + 1):
            if ciag_cyfr[i] > najlepsza_cyfra:
                najlepsza_cyfra = ciag_cyfr[i]
                indeks_najlepszej = i
        wynik += najlepsza_cyfra
        pozycja = indeks_najlepszej + 1
    return int(wynik)


suma = 0
for el in lista:
    suma += znajdz_max_liczbe(el, 12)
print(suma)
