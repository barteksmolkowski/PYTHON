from typing import TypeAlias

import numpy as np
import pytest

from features.edges import Prewitt, Sobel

Mtx: TypeAlias = np.ndarray


class TestEdgeDetection:
    @pytest.fixture
    def matrix_2d(self) -> Mtx:
        return np.random.rand(10, 10)

    @pytest.fixture
    def matrix_3d(self) -> Mtx:
        return np.random.rand(10, 10, 3)

    def test_sobel_processing(self, matrix_2d):
        detector = Sobel()
        result = detector.apply(matrix_2d)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10)

    def test_prewitt_processing(self, matrix_2d):
        detector = Prewitt()
        result = detector.apply(matrix_2d)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10)

    def test_sobel_invalid_dim_gate(self, matrix_3d):
        detector = Sobel()
        result = detector.apply(matrix_3d)
        assert result.ndim == 3

    def test_prewitt_invalid_dim_gate(self, matrix_3d):
        detector = Prewitt()
        result = detector.apply(matrix_3d)
        assert result.ndim == 3


""" Context: Przeprowadź refaktoryzację wklejonego niżej testu, wydzielając logikę współdzieloną do pliku conftest.py zgodnie z zasadami pytest fixtures. Zadania: Analiza conftest.py: Zidentyfikuj powtarzalne elementy (inicjalizacje klas jak CacheManager, przygotowanie danych MtxList, konfigurację loggera lub mocki) i stwórz z nich @pytest.fixture. Refaktoryzacja testu: Przepisz test tak, aby nie używał setup_method ani self. Ma korzystać z fixture'ów wstrzykniętych przez argumenty funkcji. Zachowanie logowania: W fixture'ach i testach zachowaj logowanie punktów decyzyjnych zgodnie z naszymi poprzednimi ustaleniami (ENG, [method_name], DEBUG/INFO/WARNING).Output format: Podaj mi wynik w dwóch wyraźnych sekcjach:DO DODANIA W CONFTEST.PY: (Kod nowych fixture'ów).POCHODNY PLIK TESTOWY: (Oczyszczony i skrócony kod testu).Dodatkowe wytyczne:Używaj scope="function" lub scope="class" zależnie od kosztu tworzenia obiektu.Importy Mtx, MtxList oraz dekoratory @silent mają zostać tam, gdzie są niezbędne.Nie pisz zbędnych komentarzy – kod ma być czysty i gotowy do wklejenia.Cel końcowy: Przygotowanie modularnej bazy,którą na końcu wspólnie zeskaleujemy i uprościmy w jednym pliku conftest.py.
"""
