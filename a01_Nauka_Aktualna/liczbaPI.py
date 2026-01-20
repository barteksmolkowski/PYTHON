from time import perf_counter
from functools import wraps

def czasFunkcjiOgolny(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        wynik = func(*args, **kwargs)
        koniec = perf_counter()
        czas = koniec - start
        print(f"Czas funkcji {func.__name__}: {czas:.8f} s")
        return wynik, czas
    return wrapper

def dodawanie(a: int| str, b: int| str, naRaz: int) -> str:
    a = int(a) if isinstance(a, str) else a
    b = int(b) if isinstance(b, str) else b
    wynik = ""
    carry = 0
    
    while a or b:
        aN = int(a[-naRaz:]) if a else 0
        bN = int(b[-naRaz:]) if b else 0

        s = aN + bN + carry
        base = 10 ** naRaz

        part = str(s % base).zfill(naRaz)
        carry = s // base

        wynik = part + wynik

        a = a[:-naRaz] if len(a) > naRaz else ""
        b = b[:-naRaz] if len(b) > naRaz else ""

    if carry:
        wynik = str(carry) + wynik

    return wynik.lstrip("0") or "0"

def mnozenie(lista: list[str | int], naRaz: int) -> str:
    sumaWynik = "1"
    base = 10 ** naRaz

    nLista = []
    for el in lista:
        if isinstance(el, int):
            nLista.append(str(el))
        else:
            nLista.append(el)

    for num_str in nLista:
        A = [int(num_str[max(i - naRaz, 0):i]) for i in range(len(num_str), 0, -naRaz)]
        B = [int(sumaWynik[max(i - naRaz, 0):i]) for i in range(len(sumaWynik), 0, -naRaz)]

        wynik = [0] * (len(A) + len(B))

        for i in range(len(A)):
            for j in range(len(B)):
                wynik[i + j] += A[i] * B[j]

        carry = 0
        for k in range(len(wynik)):
            wynik[k] += carry
            carry = wynik[k] // base
            wynik[k] = wynik[k] % base

        if carry:
            wynik.append(carry)

        wynik_str = str(wynik[-1]) + ''.join(str(x).zfill(naRaz) for x in reversed(wynik[:-1]))

        sumaWynik = wynik_str.lstrip("0") or "0"

    return sumaWynik

def silnia(liczba: int | str):
    liczba = int(liczba) if isinstance(liczba, str) else liczba
    liczby = [i for i in range(1, liczba + 1)]
    wynik, _ = mnozenie(liczby, 10)
    return wynik

def potega(a: int|str, b: int|str, naRaz) -> str:# np 3**100= 3*3*3 (100 razy)
    wynik = 1
    for _ in range(int(b)):
        wynik = mnozenie([a, wynik], naRaz)
    return wynik

def dzielenie(a, b, poPrzecinku):
    0

def pierwiastek(liczba, pierwiastkiPoPrzecinku):
    0

@czasFunkcjiOgolny
def wzorChudnovskyego(wartosci: int, poPrzecinku: int, wynikPoPrzecinku: int, naRaz: int):
    pierwszyDzielnik = dzielenie(pierwiastek(10005, poPrzecinku), 4270934400, poPrzecinku)
    suma = 0
    for k in range(1, wartosci + 1):

        pierwszaPotega = potega(-1, k, naRaz)

        goraUlam1 = silnia(mnozenie([6, k], naRaz))
        goraUlam2 = dodawanie(13591409, 545140134 * k, naRaz)
        dolUlam1 = silnia(mnozenie([3, k], naRaz))
        dolUlam2 = potega(silnia(k), 3, naRaz)
        dolUlam3 = potega(640320, mnozenie([3, k], naRaz))

        drugaCzesc = dzielenie(mnozenie([goraUlam1, goraUlam2], naRaz), mnozenie([dolUlam1, dolUlam2, dolUlam3], poPrzecinku))
        wartosc = mnozenie([pierwszaPotega, drugaCzesc], naRaz)
        suma = dodawanie(suma, wartosc)
    
    wynik = dzielenie(1, mnozenie([pierwszyDzielnik, wynik]), poPrzecinku)
    return wynik



print(potega(10, 10, 10))
wynik = silnia(10)
print(f"wynik: {wynik}\nlen: {len(wynik)}")
