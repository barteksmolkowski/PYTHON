import sys

plik = []
for linia in sys.stdin:
    linia = linia.strip()
    if not linia:
        break
    plik.append(linia)

lista, liczby = [], []
for linia in plik:
    if "-" in linia:
        l1, l2 = linia.split("-")
        l1, l2 = int(l1), int(l2)
        lista.append((l1, l2))
    else:
        try:
            liczby.append(int(linia))
        except:
            continue

print(lista)
print(liczby)

wynik = 0
for i in liczby:
    for (od_, do_) in lista:
        if od_ <= i <= do_:
            wynik += 1
            break
print(wynik)
