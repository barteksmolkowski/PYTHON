from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np

from a00_System_Baza.baza_nauki import __BazaNauki__


class podstawyProtocol(Protocol):
    def podstawy(self): ...


class podstawy(__BazaNauki__):
    def podstawy(self):
        plt.plot([5], [7], "ro")
        plt.plot([12], [4], "b^")

        xstart, xend, ystart, yend = 0, 13, 0, 8
        plt.axis([xstart, xend, ystart, yend])

        linijki = np.arange(0, 14, 1)
        plt.xticks(linijki)
        plt.yticks(linijki)

        plt.show()

    def wykresy(self):
        x = np.linspace(1, 10, 100)

        y1 = np.ones_like(x)
        y2 = 2 * x
        y3 = x**2
        y4 = np.sin(x) * 10
        y5 = x**1.5
        y6 = np.log(x) * 10
        y7 = 100 / x
        y8 = np.cos(x) * 5
        y9 = (x - 5) ** 3
        y10 = np.sqrt(x) * 20

        plt.plot(x, y1, label="y=1", lw=2)
        plt.plot(x, y2, label="y=2x", lw=2)
        plt.plot(x, y3, label="y=x^2", lw=2)
        plt.plot(x, y4, "r-", label="y=sin(x)*10", lw=4)
        plt.plot(x, y5, label="y=x^1.5", linestyle=":")
        plt.plot(x, y6, label="y=10*ln(x)")
        plt.plot(x, y7, label="y=100/x")
        plt.plot(x, y8, label="y=5*cos(x)")
        plt.plot(x, y9, label="y=(x-5)^3")
        plt.plot(x, y10, label="y=20*sqrt(x)", linestyle="-.")

        plt.yscale("linear")
        plt.xscale("linear")

        plt.xlabel("Oś X (Argumenty)")
        plt.ylabel("Oś Y (Wartości)", loc="bottom")

        plt.title("Laboratorium Wykresów 2026 - 10 Funkcji")

        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="best", fontsize=15)

        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    podstawy(aktywne=True, metody=["wykresy"])
