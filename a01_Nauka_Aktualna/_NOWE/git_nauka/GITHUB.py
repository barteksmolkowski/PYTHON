from typing import Protocol

from a01_Nauka_Aktualna import __BazaNauki__


class GitBasic_Protocol(Protocol):
    def version(self) -> None: ...
    def config_global(self) -> None: ...
    def init(self) -> None: ...
    def status(self) -> None: ...
    def add(self) -> None: ...
    def commit(self) -> None: ...
    def push(self) -> None: ...
    def pull(self) -> None: ...
    def restore_checkout_switch(self) -> None: ...


class GitDiff_Protocol(Protocol):
    def o_hash_importance(self) -> None: ...

    def diff_standard(self) -> None: ...
    def diff_stat(self) -> None: ...
    def diff_shortstat(self) -> None: ...
    def diff_patch(self) -> None: ...
    def diff_cached(self) -> None: ...
    def diff_kumulacja(self) -> None: ...

    def diff_commits(self) -> None: ...
    def diff_head_shortcuts(self) -> None: ...
    def diff_head_relative(self) -> None: ...
    def diff_commits_selective(self) -> None: ...
    def diff_separator(self) -> None: ...


class GitLog_Protocol(Protocol):
    def basic_log(self) -> None: ...
    def file_analysis_log(self) -> None: ...
    def search_log(self) -> None: ...
    def visual_log(self) -> None: ...
    def log_all_branches(self) -> None: ...
    def log_decorate(self) -> None: ...
    def config_call_alias_ll(self) -> None: ...
    def config_alias_global(self) -> None: ...
    def call_alias_ll(self) -> None: ...
    def log_graph(self) -> None: ...


class GitUndo_Protocol(Protocol):
    def quick_commit(self) -> None: ...
    def revert_changes(self) -> None: ...
    def revert_no_commit(self) -> None: ...
    def restore_file_from_hash(self) -> None: ...
    def reset_soft(self) -> None: ...
    def reset_mixed(self) -> None: ...
    def reset_hard(self) -> None: ...
    def checkout_file_from_hash(self) -> None: ...


class GitRefactor_Protocol(Protocol):
    def commit_amend(self) -> None: ...
    def rebase_interactive(self) -> None: ...
    def rebase_pick(self) -> None: ...
    def rebase_reword(self) -> None: ...
    def rebase_squash(self) -> None: ...
    def rebase_fixup(self) -> None: ...
    def rebase_drop(self) -> None: ...


class GitBranchMerge_Protocol(Protocol):
    def merge_standard(self) -> None: ...
    def merge_with_msg(self) -> None: ...
    def resolve_conflicts(self) -> None: ...
    def branch_move(self) -> None: ...
    def branch_list_merged(self) -> None: ...
    def branch_list_unmerged(self) -> None: ...
    def branch_delete_safe(self) -> None: ...
    def branch_delete_force(self) -> None: ...
    def branch_from_hash(self) -> None: ...


class GitBasic(__BazaNauki__, GitBasic_Protocol):
    opis_menu = "version, config_global, init, status, add, commit, push, pull, restore_checkout_switch"

    """
    version - git --version
    config_global - git config --global
    init - git init
    status - git status
    add - git add .
    commit - git commit -m
    push - git push
    pull - git pull
    restore_checkout_switch - git restore / git checkout / git switch
    """

    def version(self):
        print("Wersja gita:\n-> git --version")

    def config_global(self):
        print("Konfiguracja globalna:\n-> git config --global user.name 'Nazwa'")

    def init(self):
        print("Inicjalizacja repozytorium:\n-> git init")

    def status(self):
        print("Status plików:\n-> git status")

    def add(self):
        print("Dodawanie do staging:\n-> git add .")

    def commit(self):
        print("Zapisanie zmian:\n-> git commit -m 'opis'")

    def push(self):
        print("Wysyłka na serwer:\n-> git push origin main")

    def pull(self):
        print("Pobieranie zmian:\n-> git pull")

    def restore_checkout_switch(self):
        print(
            f"Cofanie i podróże w czasie:\n-> git restore <plik>     # Przywraca plik/cofa usunięcie\n-> git checkout <hash>    # Przejście do konkretnej wersji\n-> git switch -{" " * 11}# Powrót na główną gałąź"
        )


class GitDiff(__BazaNauki__, GitDiff_Protocol):
    opis_menu = "o_hash_importance, diff_standard, diff_stat, diff_shortstat, diff_patch, diff_cached, diff_kumulacja, diff_commits, diff_head_shortcuts, diff_head_relative, diff_commits_selective, diff_separator"


    """
    o_hash_importance - info o haszach
    diff_standard - git diff
    diff_stat - git diff --stat
    diff_shortstat - git diff --shortstat
    diff_patch - git diff -p --color
    diff_cached - git diff --cached
    diff_kumulacja - git diff --stat --cached
    diff_commits - git diff [h1] [h2] --stat
    diff_head_shortcuts - git diff @^ @ / HEAD~n..
    diff_head_relative - git diff HEAD~n HEAD --stat/--name-only
    diff_commits_selective - git diff [h1] [h2] -- [path]
    diff_separator - git diff -- [path]
    """

    def o_hash_importance(self):
        print(
            "WAŻNE: Hasze (np. f6f8988) są niezbędne do porównywania wersji z przeszłości!"
        )

    def diff_standard(self):
        print("Standardowe porównanie:\n-> git diff")

    def diff_stat(self):
        print("Statystyka zmian (linie +/-):\n-> git diff --stat")

    def diff_shortstat(self):
        print("Krótkie podsumowanie:\n-> git diff --shortstat")

    def diff_patch(self):
        print("Pełny podgląd (format patch) z kolorem:\n-> git diff -p --color")

    def diff_cached(self):
        print("Analiza plików DODANYCH (staging):\n-> git diff --cached")

    def diff_kumulacja(self):
        print("KUMULACJA FLAG (Łączenie minusów):\n-> git diff --stat --cached")

    def diff_commits(self):
        print("Porównanie historii między haszami:\n-> git diff <hash1> <hash2> --stat")

    def diff_head_shortcuts(self):
        print(
            "Skróty HEAD (nawigacja 2026):\n-> git diff @^ @     # @ to HEAD, ^ to poprzedni commit\n-> git diff HEAD~3.. # od 3 commitów temu do teraz"
        )

    def diff_head_relative(self):
        print(
            "Porównanie relatywne (HEAD~n):\n-> git diff HEAD~2 HEAD --stat\n-> git diff HEAD~5 HEAD --name-only"
        )

    def diff_commits_selective(self):
        print(
            "Selektywne porównanie (hasze + separator):\n-> git diff <hash1> <hash2> -- <sciezka>"
        )

    def diff_separator(self):
        print(
            "Separator (--) - skupienie na ścieżce (ignoruje branche):\n-> git diff -- <sciezka/do/pliku>"
        )


class GitLog(__BazaNauki__, GitLog_Protocol):
    opis_menu = "basic_log, file_analysis_log, search_log, visual_log, log_all_branches, log_decorate, config_call_alias_ll, config_alias_global, call_alias_ll, log_graph"


    """
    basic_log - git log --oneline (-n)
    file_analysis_log - git log -p / --stat
    search_log - git log --grep / -S
    visual_log - git log --graph --all
    log_all_branches - git log --all (wszystkie branche)
    log_decorate - git log --decorate (tagi/head)
    config_call_alias_ll - ustawienie i opis aliasu ll
    config_alias_global - globalny alias xyz
    call_alias_ll - wywołanie git ll
    log_graph - proste rysowanie grafu
    """

    def basic_log(self):
        print(
            "Podstawowa historia:\n-> git log --oneline     # Hasz + Opis\n-> git log --oneline -5  # Ostatnie 5 zmian"
        )

    def file_analysis_log(self):
        print(
            "Analiza zmian w plikach:\n-> git log -p      # Co dokładnie zmieniło się w kodzie\n-> git log --stat  # Statystyka zmienionych plików"
        )

    def search_log(self):
        print(
            "Przeszukiwanie historii:\n-> git log --grep='wiadomosc' # Szukaj po opisie\n-> git log -S 'kod' -p        # 'Pickaxe' - szukaj frazy w KODZIE"
        )

    def visual_log(self):
        print(
            "Widok graficzny (Full):\n-> git log --oneline --graph --all # Drzewo wszystkich gałęzi"
        )

    def log_all_branches(self):
        print(
            "Logowanie wszystkich gałęzi:\n-> git log --all # Wyświetla commity ze wszystkich branchy, nie tylko obecnego"
        )

    def log_decorate(self):
        print(
            "Informacje o tagach i HEAD:\n-> git log --decorate # Pokazuje wskaźniki gałęzi i tagów przy commitach"
        )

    def config_call_alias_ll(self):
        print(
            "Konfiguracja aliasu:\n-> git config alias.ll 'log --oneline --all --decorate --graph'\n# Pozwala używać 'git ll' zamiast długiej komendy"
        )

    def config_alias_global(self):
        print(
            "Definiowanie aliasu globalnego:\n-> git config --global alias.xyz 'string' # Alias dostępny we WSZYSTKICH projektach"
        )

    def call_alias_ll(self):
        print(
            "Wywołanie aliasu:\n-> git ll # Skrót do pełnego widoku graficznego (musi być wcześniej skonfigurowany)"
        )

    def log_graph(self):
        print(
            "Prosty graf:\n-> git log --graph # Rysuje drzewo zmian dla obecnej gałęzi"
        )


class GitUndo(__BazaNauki__, GitUndo_Protocol):
    opis_menu = "quick_commit, revert_changes, revert_no_commit, restore_file_from_hash, reset_soft, reset_mixed, reset_hard, checkout_file_from_hash"


    """
    quick_commit - git commit -am
    revert_changes - git revert
    revert_no_commit - git revert -n
    restore_file_from_hash - git restore -s
    reset_soft - git reset --soft
    reset_mixed - git reset --mixed
    reset_hard - git reset --hard
    checkout_file_from_hash - git checkout [hash] -- [file]
    """

    def quick_commit(self):
        print(
            "Szybkie zatwierdzenie wszystkich zmodyfikowanych plików bez ręcznego dodawania:\n-> git commit -a -m 'message'"
        )

    def revert_changes(self):
        print(
            "Bezpieczne wycofanie zmian z konkretnego zapisu poprzez stworzenie nowego commita:\n-> git revert <hash>"
        )

    def revert_no_commit(self):
        print(
            "Wycofanie zmian z dwóch ostatnich zapisów do poczekalni bez automatycznego tworzenia commita:\n-> git revert --no-commit HEAD~2"
        )

    def restore_file_from_hash(self):
        print(
            "Przywrócenie wybranego pliku do stanu, w jakim był w konkretnym momencie historii:\n-> git restore --source <hash> -- <plik>"
        )

    def reset_soft(self):
        print(
            "Cofnięcie zapisu do historii przy jednoczesnym zachowaniu wszystkich zmian w kodzie:\n-> git reset <hash> --soft"
        )

    def reset_mixed(self):
        print(
            "cofnięcie commita i wyczyszczenie poczekalni, ale zmiany zostają w kodzie (tryb domyślny):\n-> git reset <hash> --mixed"
        )

    def reset_hard(self):
        print(
            "całkowite skasowanie zmian i powrót do czystego stanu z danego hasha (uważaj, tracisz kod!):\n-> git reset <hash> --hard"
        )

    def checkout_file_from_hash(self):
        print(
            "nadpisanie konkretnego pliku wersją z innego zapisu/hasha:\n-> git checkout <hash> -- <plik>"
        )


class GitRefactor(__BazaNauki__, GitRefactor_Protocol):
    opis_menu = "quick_commit, revert_changes, revert_no_commit, restore_file_from_hash, reset_soft, reset_mixed, reset_hard, checkout_file_from_hash"


    """
    quick_commit - git commit -am
    revert_changes - git revert
    revert_no_commit - git revert -n
    restore_file_from_hash - git restore -s
    reset_soft - git reset --soft
    reset_mixed - git reset --mixed
    reset_hard - git reset --hard
    checkout_file_from_hash - git checkout [hash] -- [file]
    """

    def commit_amend(self):
        print(
            "szybka poprawka ostatniego zapisu (zmiana nazwy lub dodanie zapomnianych plików):\n-> git commit --amend -m 'nowa_nazwa'"
        )

    def rebase_interactive(self):
        print(
            "wejście w tryb interaktywnej edycji ostatnich n zapisów:\n-> git rebase -i HEAD~3"
        )

    def rebase_pick(self):
        print(
            "pozostawienie zapisu bez żadnych zmian (standard w menu rebase):\n-> p / pick"
        )

    def rebase_reword(self):
        print(
            "zatrzymanie procesu, aby zmienić tylko tekst opisu danego zapisu:\n-> r / reword"
        )

    def rebase_squash(self):
        print(
            "połączenie zapisu z poprzednim (scala kod i pozwala edytować wspólny opis):\n-> s / squash"
        )

    def rebase_fixup(self):
        print(
            "połączenie zapisu z poprzednim po cichu (zostawia tylko opis starszego zapisu):\n-> f / fixup"
        )

    def rebase_drop(self):
        print(
            "całkowite usunięcie wybranego zapisu wraz z jego zmianami w kodzie:\n-> d / drop"
        )


class GitBranchMerge(__BazaNauki__, GitBranchMerge_Protocol):
    opis_menu = "merge_standard, merge_with_msg, resolve_conflicts, branch_move, branch_list_merged, branch_list_unmerged, branch_delete_safe, branch_delete_force, branch_from_hash"

    """
    merge_standard - git merge nazwa (scala gałęzie)
    merge_with_msg - git merge -m (własny komentarz)
    resolve_conflicts - porada (Merge Editor / Ręcznie)
    branch_move - git branch --move (zmiana nazwy)
    branch_list_merged - git branch --merged (złączone)
    branch_list_unmerged - git branch --no-merged (niezłączone)
    branch_delete_safe - git branch -d (bezpieczne usuwanie)
    branch_delete_force - git branch -D (siłowe usuwanie)
    branch_from_hash - git branch nazwa hasz (wskrzeszanie)
    """

    def merge_standard(self):
        print(
            "Łączenie gałęzi:\n-> git merge 'nazwa_brancha' # Scala wskazany branch do aktualnego"
        )

    def merge_with_msg(self):
        print(
            "Łączenie z własnym opisem:\n-> git merge 'nazwa' -m 'komentarz' # Nadpisuje domyślny komunikat"
        )

    def resolve_conflicts(self):
        print(
            "Rozwiązywanie konfliktów:\n-> Ręcznie (Zalecane): Usuń <<<< ==== >>>> z pliku, zostaw kod i zrób commit"
        )

    def branch_move(self):
        print(
            "Zmiana nazwy gałęzi:\n-> git branch --move 'stara' 'nowa' # Przemianowuje brancha"
        )

    def branch_from_hash(self):
        print(
            "Wskrzeszanie gałęzi:\n-> git branch 'nowa_nazwa' <hash> # Tworzy brancha w miejscu starego commita"
        )

    def branch_list_merged(self):
        print(
            "Gałęzie już złączone:\n-> git branch --merged # Te, które można bezpiecznie usunąć"
        )

    def branch_list_unmerged(self):
        print(
            "Gałęzie niezłączone:\n-> git branch --no-merged # Gałęzie z nowym kodem, jeszcze nie w main"
        )

    def branch_delete_safe(self):
        print(
            "Bezpieczne usuwanie:\n-> git branch -d 'nazwa' # Tylko jeśli branch był złączony"
        )

    def branch_delete_force(self):
        print(
            "Siłowe usuwanie:\n-> git branch -D 'nazwa' # Usuwa nawet niezłączone (-D to --delete --force)"
        )


if __name__ == "__main__":
    moje_lekcje = [GitBasic, GitDiff, GitLog, GitUndo, GitRefactor, GitBranchMerge]

    __BazaNauki__(interaktywne=True, lista_klas=moje_lekcje)
