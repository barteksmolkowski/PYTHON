import os
from typing import Protocol

from PYTHON_NAUKA import __BazaNauki__, bezpieczny_wrapper


class FileManagerProtocol(Protocol):
    def list_files_and_dirs(self): ...
    def create_directory(self): ...
    def rename_item(self): ...
    def path_info(self): ...
    def create_nested_dirs_and_file(self): ...


class FileManager(__BazaNauki__):
    @bezpieczny_wrapper
    def list_files_and_dirs(self):
        print("Sprawdzenie, czy element jest plikiem czy folderem w bieżącym katalogu:")
        for el in os.listdir("."):
            if os.path.isfile(el):
                print(f"{el} jest plikiem")
            elif os.path.isdir(el):
                print(f"{el} jest folderem")

    @bezpieczny_wrapper
    def create_directory(self):
        print("Tworzenie katalogu 'pliki' jeśli nie istnieje:")
        if not os.path.exists("pliki"):
            os.mkdir("pliki")
            print("Katalog 'pliki' został utworzony")
        else:
            print("Katalog 'pliki' już istnieje")

    @bezpieczny_wrapper
    def rename_item(self):
        print("Przykład zmiany nazwy pliku/katalogu:")
        print("Użycie: os.rename('staraNazwa', 'nowaNazwa')")

    @bezpieczny_wrapper
    def path_info(self):
        path = "pliki/01/dane.txt"
        print(f"Przykładowa ścieżka: {path}")
        print(f"dirname: {os.path.dirname(path)}")
        print(f"basename: {os.path.basename(path)}")
        print(f"abspath: {os.path.abspath(path)}")

    @bezpieczny_wrapper
    def create_nested_dirs_and_file(self):
        path = "pliki/01/dane.txt"
        print(f"Tworzenie zagnieżdżonych katalogów do: {os.path.dirname(path)}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as plik:
            print(f"stworzono plik: {path}")


if __name__ == "__main__":
    FileManager(True)
