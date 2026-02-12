from typing import Protocol, runtime_checkable

import numpy as np
from numba import jit, njit

from a00_System_Baza.baza_nauki import __BazaNauki__


@runtime_checkable
class ISilnikJIT(Protocol):
    """
    optymalizacja_podstawy_njit                  -> @njit
    tryby_kompilacji_njit_vs_jit                 -> @njit vs @jit
    dlaczego_zawsze_njit                         -> @njit
    inspekcja_niskopoziomowa_metoda_inspect_asm  -> po @njit metoda.inspect_asm()
    """

    def optymalizacja_podstawy_njit(self) -> None: ...
    def tryby_kompilacji_njit_vs_jit(self) -> None: ...
    def dlaczego_zawsze_njit(self) -> None: ...
    def inspekcja_niskopoziomowa_metoda_inspect_asm(self) -> None: ...


class SilnikObliczeniowyJIT(__BazaNauki__):
    def optymalizacja_jit_numba_podstawy(self):
        print(f"\n=> PODSTAWY OPTYMALIZACJI JIT (NUMBA) <=")

        @njit
        def fast_sum_logic(matrix, threshold):
            count = 0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if matrix[i, j] > threshold:
                        count += 1
            return count

        test_m = np.random.randint(0, 10, (100, 100))
        threshold_val = 5

        result = fast_sum_logic(test_m, threshold_val)
        print(f"Liczba elementów > {threshold_val} (wynik Numba): {result}")

    def tryby_kompilacji_njit_vs_jit(self):
        """
        Różnica między trybem nopython (wydajność) a object mode (kompatybilność).
        """
        print(f"\n=> TRYBY KOMPILACJI: NJIT VS JIT <=")

        # @njit (skrót od @jit(nopython=True))
        # Wymaga, aby cały kod mógł być skompilowany do postaci maszynowej.
        # To "Złoty Standard" 2026: GWARANCJA wydajności lub błąd kompilacji.
        @njit
        def tylko_maszynowo(A):
            return A.sum()

        @jit(nopython=False)
        def tryb_mieszany(A):
            print(
                "Logowanie: ", A[0]
            )  # print() wymusza spadek do wolnego trybu Pythona
            return A * 2

        test_data = np.array([10, 20, 30])

        print(f"Wynik @njit: {tylko_maszynowo(test_data)}")
        print(f"Wynik @jit (object mode): {tryb_mieszany(test_data)}")

    def dlaczego_zawsze_njit(self):
        """
        Zasada 'Fail Fast' w inżynierii oprogramowania.
        """
        print(f"\n=> ZASADA PROJEKTOWA: DLACZEGO @NJIT? <=")
        # Używając @njit masz pewność, że Twój kod jest szybki.
        # Jeśli Numba zgłosi błąd kompilacji, to informuje Cię, że dana operacja
        # spowolniłaby cały proces. Wymusza to pisanie czystego, wydajnego kodu.

        print(
            "Rekomendacja 2026: Zawsze używaj @njit. Unikaj @jit(nopython=False), które ukrywa problemy z wydajnością."
        )

    def inspekcja_niskopoziomowa(self):
        """
        Analiza kodu asemblerowego (LLVM IR) wygenerowanego przez JIT.
        """
        print(f"\n=> INSPEKCJA KODU MASZYNOWEGO (Dla zaawansowanych) <=")

        # Wymaga wcześniejszego wywołania funkcji, aby kompilacja zaszła
        # fast_sum_logic.inspect_asm(fast_sum_logic.signatures[0])
        print(
            "Notatka: Funkcja .inspect_asm() pozwala sprawdzić, jak Numba zoptymalizowała pętle do instrukcji CPU (np. SIMD)."
        )


if __name__ == "__main__":
    SilnikObliczeniowyJIT(aktywne=True)
