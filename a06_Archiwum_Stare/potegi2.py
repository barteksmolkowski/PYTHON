import sys

sys.set_int_max_str_digits(1000000)

x = int(input("Podaj liczbę: "))
a = 1
tabela = []
megaTab = []
for i in range(x + 1):
    a *= 1024
    numer = (1 + i) * 100
    wynik = round(a)
    txtwynik = str(wynik)
    if int(txtwynik[:2]) > 98:
        megaTab.append((numer, int(txtwynik[:10])))
print(tabela)
print(str(2**100))
for i in range(len(megaTab)):
    print(megaTab[i])
