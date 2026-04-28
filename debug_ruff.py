import json
import os
import subprocess
import sys


def check_step(name, condition, error_msg):
    if condition:
        print(f"✅ {name}")
        return True
    else:
        print(f"❌ {name} -> {error_msg}")
        return False


print("🔍 DIAGNOSTYKA AUTOPORZĄDKOWANIA (RUFF + CMD+S)\n")

# 1. Sprawdzenie czy Ruff jest w systemie
ruff_version = subprocess.getoutput("ruff --version")
check_step(
    "Biblioteka Ruff w systemie",
    "ruff" in ruff_version.lower(),
    "Zainstaluj ruff komendą: python3 -m pip install ruff",
)

# 2. Sprawdzenie czy VS Code widzi rozszerzenie
installed_exts = subprocess.getoutput("code --list-extensions")
check_step(
    "Rozszerzenie VS Code (astral-sh.ruff)",
    "astral-sh.ruff" in installed_exts,
    "Brak rozszerzenia 'Ruff' od Astral Software w VS Code!",
)

# 3. Analiza pliku settings.json (User)
settings_path = os.path.expanduser(
    "~/Library/Application Support/Code/User/settings.json"
)
try:
    with open(settings_path, "r") as f:
        # Usuwamy komentarze z JSONa, jeśli istnieją, żeby go sparsować
        lines = [l for l in f.readlines() if not l.strip().startswith("//")]
        data = json.loads("".join(lines))

    python_settings = data.get("[python]", {})

    # Testy konkretnych kluczy
    check_step(
        "Ustawienie formatOnSave",
        data.get("editor.formatOnSave") is True
        or python_settings.get("editor.formatOnSave") is True,
        "Musisz dodać 'editor.formatOnSave': true",
    )

    check_step(
        "Default Formatter (Ruff)",
        python_settings.get("editor.defaultFormatter") == "astral-sh.ruff",
        "Błędny formatter! Powinien być 'astral-sh.ruff'",
    )

    actions = python_settings.get("editor.codeActionsOnSave", {})
    check_step(
        "Akcja OrganizeImports (Ruff)",
        actions.get("source.organizeImports.ruff") == "always",
        "Brakuje 'source.organizeImports.ruff': 'always'",
    )

    check_step(
        "Akcja FixAll (Ruff)",
        actions.get("source.fixAll.ruff") == "always",
        "Brakuje 'source.fixAll.ruff': 'always'",
    )

except Exception as e:
    print(f"❌ BŁĄD ODCZYTU SETTINGS.JSON: {e}")

print("\n💡 CO ZROBIĆ, JEŚLI WSZYSTKO JEST NA ZIELONO, A NADAL NIE DZIAŁA?")
print(
    "1. Sprawdź, czy nie masz pliku .vscode/settings.json w folderze projektu, który NADPISUJE te ustawienia (Plik B)."
)
print(
    "2. Upewnij się, że w prawym dolnym rogu VS Code masz wybrany ten sam Interpreter Pythona, co w terminalu."
)
print(
    "3. Spróbuj skrótu Shift + Option + F – jeśli to nie sformatuje kodu, Ruff ma problem z samym plikiem (np. błąd składni)."
)
