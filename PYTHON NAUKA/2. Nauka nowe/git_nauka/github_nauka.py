from typing import Protocol

from baza_nauki import __BazaNauki__


class GitBasicCommands(Protocol):
    def version(self): ...
    def config_global(self): ...
    def init(self): ...
    def status(self): ...
    def add(self): ...
    def commit(self): ...
    def push(self): ...
    def pull(self): ...

class GitDiffCommands(Protocol):
    def diff(self): ...
    def diff_stat(self): ...
    def diff_shortstat(self): ...
    def diff_patch(self): ...

class GitLogCommands(Protocol):
    def log(self): ...


class GitBasic:
    def version(self):
        print("git --version\n- Pokazuje zainstalowaną wersję Gita.\n\nPrzykład:\ngit --version")

    def config_global(self):
        print("Konfiguracja globalna (podpis autora):\n"
              "git config --global user.name 'barteksmolkowski'\n"
              "git config --global user.email 'bartek.smolkowski@gmail.com'")

    def init(self):
        print("Inicjalizacja nowego, niezależnego repozytorium:\n"
              "git init")

    def status(self):
        print("Sprawdzenie statusu repozytorium (co zmienione, co gotowe do commita):\n"
              "git status")

    def add(self):
        print("Dodanie plików do staging area (przygotowanie do commita):\n"
              "git add <nazwa_pliku>\n"
              "git add .   # dodaje wszystkie zmienione pliki z aktualnego katalogu i podkatalogów")

    def commit(self):
        print("Zapisanie zmian w commicie z opisem:\n"
              "git commit -m \"Opis zmian\"")

    def push(self):
        print("Wysłanie commitów do zdalnego repozytorium:\n"
              "git push")

    def pull(self):
        print("Pobranie zmian ze zdalnego repozytorium:\n"
              "git pull")


class GitDiff:
    def diff(self):
        print("Wyświetlenie różnic między plikami (working directory vs staging):\n"
              "git diff")

    def diff_stat(self):
        print("Wyświetlenie statystyki zmian w plikach:\n"
              "git diff --stat")

    def diff_shortstat(self):
        print("Krótkie podsumowanie liczby zmian:\n"
              "git diff --shortstat")

    def diff_patch(self):
        print("Wyświetlenie zmian w formacie patch (z kontekstem funkcji/metod):\n"
              "git diff -p")


class GitLog:
    def log(self):
        print("Wyświetlenie historii commitów:\n"
              "git log\n"
              "git log --oneline   # skrócona wersja historii")
