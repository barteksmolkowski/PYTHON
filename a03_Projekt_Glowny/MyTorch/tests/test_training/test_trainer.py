import logging

import pytest

from training.trainer import Trainer

logger = logging.getLogger(__name__)


class TestTrainer:
    @pytest.fixture
    def trainer_instance(self):
        return Trainer()

    def test_train(self, trainer_instance):
        logger.info("ACTION: Testing train")

        result = trainer_instance.train(epochs=10)

        if result is not None:
            logger.error("Assertion failed: train should return None")
        assert result is None

        logger.info("SUCCESS: train verified")

    def test_evaluate(self, trainer_instance):
        logger.info("ACTION: Testing evaluate")

        result = trainer_instance.evaluate()

        if result is not None:
            logger.error("Assertion failed: evaluate should return None")
        assert result is None

        logger.info("SUCCESS: evaluate verified")


"""
Dodaj logowanie wewnątrz metod zgodnie z poniższymi zasadami:English: Wszystkie komunikaty logów w języku ang.No setup: Nie dodawaj 'import logging' ani 'logger = ...'. Użyj istniejącego obiektu 'self.logger'. Prefix format: Każdy log musi zaczynać się od nazwy metody w nawiasach: [method_name] Message. (System automatycznie doda timestamp dzięki setup_logging). Auto-wrapper awareness: Klasa jest owinięta w @class_autologger. NIE LOGUJ: wejścia do metody (start),wyjścia z metody (end/summary), czasu trwania, ogólnych bloków try-except metody.To wszystko jest już obsługiwane automatycznie. Hierarchy: INFO: Tylko kluczowe punkty milowe wewnątrz logiki (jeśli potrzebne). DEBUG: Punkty decyzyjne (wnętrze pętli, powody odrzuceń w instrukcjach 'if' wraz z wartościami zmiennych) WARNING/ERROR: Specyficzne problemy biznesowe (np. 'max_attempts reached'). Logic-first: Nie edytuj logiki kodu. Logi mają towarzyszyć instrukcjom'if', 'continue', 'break' oraz 'return'. DEBUG: Punkty decyzyjne (wnętrze pętli, powody odrzuceń w instrukcjach 'if').Zawsze uwzględniaj bieżące wartości zmiennych w logach, aby wyjaśnić, dlaczego podjęto daną decyzję
.
"""
