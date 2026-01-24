import sys


def is_invalid(num: int) -> bool:
    s = str(num)
    n = len(s)

    for k in range(1, n // 2 + 1):
        if n % k == 0:
            blok = s[:k]
            if blok * (n // k) == s:
                return True
    return False


def suma_invalid(range_str: str) -> int:
    a, b = map(int, range_str.split("-"))
    suma = 0
    for x in range(a, b + 1):
        if is_invalid(x):
            suma += x
    return suma


lista = []
for linia in sys.stdin:
    linia = linia.strip()
    if linia:
        lista.append(linia)
        break

przedzialy = lista[0].replace(" ", "").split(",")

suma = 0
for zakres in przedzialy:
    suma += suma_invalid(zakres)

print(suma)
