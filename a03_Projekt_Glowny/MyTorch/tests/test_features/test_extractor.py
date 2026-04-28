from typing import TypeAlias

import numpy as np
import pytest

from features.extractor import FeatureExtraction

Mtx: TypeAlias = np.ndarray


class TestFeatureExtraction:
    @pytest.fixture
    def valid_mtx(self) -> Mtx:
        return np.ones((28, 28))

    @pytest.fixture
    def empty_mtx(self) -> Mtx:
        return np.array([])

    @pytest.fixture
    def invalid_mtx(self) -> Mtx:
        return np.ones((28, 28, 3))

    def test_extract_edges_gate_logging(self, invalid_mtx):
        extractor = FeatureExtraction()
        result = extractor.extract_edges(invalid_mtx)
        assert result.ndim == 3

    def test_extract_features_decision_point(self, empty_mtx):
        extractor = FeatureExtraction()
        result = extractor.extract_features(empty_mtx)
        assert result == [0.0]

    def test_extract_edges_valid_flow(self, valid_mtx):
        extractor = FeatureExtraction()
        result = extractor.extract_edges(valid_mtx)
        assert result.shape == (28, 28)

    @pytest.mark.parametrize("shape", [(10,), (5, 5, 5), (1, 1, 1, 1)])
    def test_gate_blocking_various_dims(self, shape):
        extractor = FeatureExtraction()
        bad_mtx = np.zeros(shape)
        result = extractor.extract_edges(bad_mtx)
        assert result.shape == shape


""" Context: Przeprowadź refaktoryzację wklejonego niżej testu, wydzielając logikę współdzieloną do pliku conftest.py zgodnie z zasadami pytest fixtures. Zadania: Analiza conftest.py: Zidentyfikuj powtarzalne elementy (inicjalizacje klas jak CacheManager, przygotowanie danych MtxList, konfigurację loggera lub mocki) i stwórz z nich @pytest.fixture. Refaktoryzacja testu: Przepisz test tak, aby nie używał setup_method ani self. Ma korzystać z fixture'ów wstrzykniętych przez argumenty funkcji. Zachowanie logowania: W fixture'ach i testach zachowaj logowanie punktów decyzyjnych zgodnie z naszymi poprzednimi ustaleniami (ENG, [method_name], DEBUG/INFO/WARNING).Output format: Podaj mi wynik w dwóch wyraźnych sekcjach:DO DODANIA W CONFTEST.PY: (Kod nowych fixture'ów).POCHODNY PLIK TESTOWY: (Oczyszczony i skrócony kod testu).Dodatkowe wytyczne:Używaj scope="function" lub scope="class" zależnie od kosztu tworzenia obiektu.Importy Mtx, MtxList oraz dekoratory @silent mają zostać tam, gdzie są niezbędne.Nie pisz zbędnych komentarzy – kod ma być czysty i gotowy do wklejenia.Cel końcowy: Przygotowanie modularnej bazy,którą na końcu wspólnie zeskaleujemy i uprościmy w jednym pliku conftest.py.
"""
