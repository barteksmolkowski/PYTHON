import os

class GlebokoscLowerValueException(Exception):
    def __init__(self, glebokosc):
        super().__init__(f"Wartość {glebokosc} dla zmiennej Głębokość jest mniejsza od 1.")
class sciezkiNotIsStrException(Exception):
    def __init__(self, sciezki):
        super().__init__(f"Ścieżka {sciezki} nie jest pojedyńczą wartością str.")
class sciezkiNotIsFolderException(Exception):
    def __init__(self, sciezki, e):
        super().__init__(f"Błąd rozpakowania {sciezki}: {e}.")

def sprawdzDane(func):
    def wrapper(sciezki, glebokosc, tekst):
        if glebokosc <= 0: raise GlebokoscLowerValueException(glebokosc)
        if not isinstance(sciezki, str): raise sciezkiNotIsStrException(sciezki)
        
        try:
            sciezki = [os.path.join(sciezki, x) for x in os.listdir(sciezki)]
        except Exception as e:
            raise sciezkiNotIsFolderException(sciezki, e)
        
        wynik = func(sciezki, glebokosc, tekst)

        return wynik
    
    return wrapper

@sprawdzDane
def wypakujSciezki1(sciezki: str, glebokosc: int, tekst: str):
    znalezione = []
    for _ in range(glebokosc):
        nastepne = []

        if not sciezki:
            break

        print(f"Długość listy ścieżek: {len(sciezki)}")
        print(f"kawałek:{sciezki[:100]}")
        for sciezka in sciezki:
            nazwa = os.path.basename(sciezka)

            if tekst in nazwa:
                znalezione.append(sciezka)

            if os.path.isdir(sciezka):
                try:
                    for i in os.listdir(sciezka):
                        nastepne.append(os.path.join(sciezka, i))
                except PermissionError:
                    continue
                except Exception:
                    continue

        sciezki = nastepne

    return znalezione

sciezki = "C:/"
wynik = wypakujSciezki1(sciezki, 3, "python")
print("Znalezione pliki/katalogi zawierające 'test' w nazwie:")
for p in wynik[:50]:
    print(p)