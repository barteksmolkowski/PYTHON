import inspect

"""
inspect.signature() # Pobiera listę z *args w wrapperach
inspect.getsource() # Zwraca kod źródłowy funkcji jako tekst
inspect.stack() # Pokazuje całą "drabinę" wywołań (kto kogo wywołał)
"""


def dekorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# inspect.stack().line
@dekorator
def przykladowa_funkcja(a, b=10, *args):
    """To jest docstring naszej funkcji."""
    stos = inspect.stack()
    print(f"--- Kto mnie wywołał? (inspect.stack) ---")
    print(f"Wywołano z: {stos[1].function} w linii {stos[1].lineno}\n")
    return a + b


sig = inspect.signature(przykladowa_funkcja)

print(f"1. EFEKT SIGNATURE: {sig}")
print(f"2. EFEKT GETSOURCE:\n{inspect.getsource(przykladowa_funkcja)}")


def starter():
    przykladowa_funkcja(5)


starter()

import inspect
from functools import wraps


def autologger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)

        bound = sig.bind(*args, **kwargs)

        bound.apply_defaults()

        slownik_argumentow = dict(bound.arguments)

        print(f"DEBUG: {func.__name__} wywołana z: {slownik_argumentow}")

        return func(*args, **kwargs)

    return wrapper


class Robot:
    @autologger
    def idz_do(self, x, y, szybkosc="Normalna"):
        return f"Ide do {x}, {y}"


r = Robot()
r.idz_do(10, y=20)
