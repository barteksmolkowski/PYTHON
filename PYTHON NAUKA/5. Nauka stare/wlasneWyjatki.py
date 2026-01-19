class TooColdException(Exception):
    def __init__(self, temp):
        self.temp = temp
        super().__init__(f"Temperatura {temp} jest za niska!")

def stopnieNaKelvin(temp):
    if temp < -273.15:
        raise TooColdException(temp)
    return temp + 273.15

try:
    wynik = stopnieNaKelvin(-1000)
    print(wynik)
except Exception as e:
    print(f"błąd: {e}")