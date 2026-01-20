from noweos import czasFunkcjiOgolny

def __fibo__(liczba):
    if liczba <= 1:
        return 1
    return __fibo__(liczba - 1) + __fibo__(liczba - 2)

@czasFunkcjiOgolny
def fibo(liczba):
    wynik = __fibo__(liczba)
    return wynik

print(fibo(10))