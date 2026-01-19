import sys

lista = []
for linia in sys.stdin:
    linia = linia.strip()
    if not linia:
        break
    lista.append(linia)

nList = []
for linia in lista:
    l1, l2 = linia.split("-")
    l1, l2 = int(l1), int(l2)
    nList.append((l1, l2))

lista = nList

lista.sort(key=lambda x: x[0])

polaczone = []
start, end = lista[0]

for i in range(1, len(lista)):
    akt_start, akt_koniec = lista[i]
    if akt_start <= end + 1:
        koniec = max(koniec, akt_koniec)
    else:
        polaczone.append((start, koniec))
        start, koniec = akt_start, akt_koniec

polaczone.append((start, koniec))

wynik = sum(koniec - start + 1 for start, koniec in polaczone)
print(wynik)
