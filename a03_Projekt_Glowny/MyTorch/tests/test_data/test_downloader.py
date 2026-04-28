from typing import TypeAlias

import numpy as np
from common_utils import class_autologger, silent
from data.downloader import DataDownloader, DataProcessor, ProjectManager

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


@class_autologger
class TestProjectPipeline:
    def setup_method(self, *_):
        self.test_url = "http://api.test.com"
        self.dummy_data = [{"id": 1}, {"id": 2}]
        self.logger.debug(
            f"[setup_method] Test environment set for URL: {self.test_url}"
        )

    def test_downloader_fetch_logic(self):
        downloader = DataDownloader()

        data = downloader.fetch_data(self.test_url, force_reload=False)

        if not isinstance(data, (list, dict)):
            self.logger.error(
                f"[test_downloader_fetch_logic] Unexpected data type: {type(data)} for source {self.test_url}"
            )

        if len(data) == 0:
            self.logger.debug(
                f"[test_downloader_fetch_logic] Received empty dataset as expected from mock-like source."
            )

        assert isinstance(data, (list, dict))

    def test_processor_batch_execution(self):
        processor = DataProcessor()

        success = processor.process_batch(self.dummy_data, grayscale=True)

        if not success:
            self.logger.warning(
                f"[test_processor_batch_execution] Processing failed for batch of size {len(self.dummy_data)}"
            )
        else:
            self.logger.info(
                f"[test_processor_batch_execution] Successfully processed {len(self.dummy_data)} items."
            )

        assert success is True

    def test_pipeline_fast_mode_logic(self):
        pm = ProjectManager()

        result = pm.run_pipeline(self.test_url, fast_mode=True)

        if not isinstance(result, bool):
            self.logger.error(
                f"[test_pipeline_fast_mode_logic] Fast mode mismatch. Expected bool, got {type(result)}"
            )

        self.logger.debug(
            f"[test_pipeline_fast_mode_logic] Pipeline run-fast completed for {self.test_url}."
        )
        assert isinstance(result, bool)

    @silent
    def test_pipeline_standard_mode_return(self):
        pm = ProjectManager()

        result = pm.run_pipeline(self.test_url, fast_mode=False)

        if result is None:
            self.logger.warning(
                f"[test_pipeline_standard_mode_return] Standard pipeline returned None for {self.test_url}"
            )

    def test_downloader_save_operation(self):
        downloader = DataDownloader()

        save_status = downloader.save_to_disk(self.dummy_data)

        if save_status:
            self.logger.info(
                f"[test_downloader_save_operation] Disk write confirmed for {len(self.dummy_data)} items."
            )

        assert save_status is True


""" Context: Przeprowadź refaktoryzację wklejonego niżej testu, wydzielając logikę współdzieloną do pliku conftest.py zgodnie z zasadami pytest fixtures. Zadania: Analiza conftest.py: Zidentyfikuj powtarzalne elementy (inicjalizacje klas jak CacheManager, przygotowanie danych MtxList, konfigurację loggera lub mocki) i stwórz z nich @pytest.fixture. Refaktoryzacja testu: Przepisz test tak, aby nie używał setup_method ani self. Ma korzystać z fixture'ów wstrzykniętych przez argumenty funkcji. Zachowanie logowania: W fixture'ach i testach zachowaj logowanie punktów decyzyjnych zgodnie z naszymi poprzednimi ustaleniami (ENG, [method_name], DEBUG/INFO/WARNING).Output format: Podaj mi wynik w dwóch wyraźnych sekcjach:DO DODANIA W CONFTEST.PY: (Kod nowych fixture'ów).POCHODNY PLIK TESTOWY: (Oczyszczony i skrócony kod testu).Dodatkowe wytyczne:Używaj scope="function" lub scope="class" zależnie od kosztu tworzenia obiektu.Importy Mtx, MtxList oraz dekoratory @silent mają zostać tam, gdzie są niezbędne.Nie pisz zbędnych komentarzy – kod ma być czysty i gotowy do wklejenia.Cel końcowy: Przygotowanie modularnej bazy,którą na końcu wspólnie zeskaleujemy i uprościmy w jednym pliku conftest.py.
"""
