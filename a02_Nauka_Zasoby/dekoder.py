import time


def czasFunkcji(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        wynik = func(*args, **kwargs)
        koniec = time.time()
        czas = koniec - start
        print(f"funkcja: {func.__name__}")  # jak wyciągnąć nazwe
        print(f"czas: {czas}")
        return wynik

    return wrapper


@czasFunkcji
def mnozenie(a, b):
    return a * b


print(mnozenie(10**3, 99))
