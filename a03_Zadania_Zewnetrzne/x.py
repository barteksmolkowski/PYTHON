import sys

lista = []
for linia in sys.stdin:
    linia = linia.strip()
    if not linia:
        break
    lista.append(linia)

krag = 50
licznik_zer = 0

for linia in lista:
    kierunek = linia[0]
    ilosc = int(linia[1:])
    print(f"krag: {krag}, linia: {linia}")
    if kierunek == "R":
        for i in range(ilosc):
            krag = (krag + 1) % 100
            if krag == 0:
                licznik_zer += 1
                print(f"krag: {krag}, linia: {linia} ZERO")
    else:
        for i in range(ilosc):
            krag = (krag - 1) % 100
            if krag == 0:
                licznik_zer += 1
                print(f"krag: {krag}, linia: {linia} ZERO")

print(licznik_zer)