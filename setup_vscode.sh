#!/bin/bash

echo "🚀 ROZPOCZYNAMY KONFIGURACJĘ VS CODE..."

# 1. SPRAWDZENIE PYTHONA I RUFFA
if ! command -v python3 &> /dev/null; then
    echo "❌ BŁĄD: Python3 nie jest zainstalowany. Pobierz go najpierw!"
else
    echo "✅ Python3 wykryty."
    python3 -m pip install ruff --quiet
    echo "✅ Biblioteka ruff sprawdzona/zainstalowana."
fi

# 2. INSTALACJA ROZSZERZEŃ VS CODE (Tylko brakujące)
EXTENSIONS=(
    "ms-python.python"
    "ms-python.vscode-pylance"
    "astral-sh.ruff"
    "njpwerner.autodocstring"
    "ms-mssql.mssql"
)

INSTALLED_EXTS=$(code --list-extensions)

for ext in "${EXTENSIONS[@]}"; do
    if echo "$INSTALLED_EXTS" | grep -qi "$ext"; then
        echo "✅ Rozszerzenie $ext jest już zainstalowane."
    else
        echo "📦 Instaluję brakujące: $ext..."
        code --install-extension "$ext"
    fi
done

# 3. PRZYGOTOWANIE PLIKU SETTINGS.JSON
USER_SETTINGS_PATH="$HOME/Library/Application Support/Code/User/settings.json"
mkdir -p "$(dirname "$USER_SETTINGS_PATH")"

# TUTAJ WKLEJ SWOJĄ TREŚĆ MIĘDZY EOF a EOF
cat <<EOF > "$USER_SETTINGS_PATH"
{
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    "editor.detectIndentation": false,
    "explorer.confirmDragAndDrop": false,
    "explorer.confirmDelete": false,
    "terminal.integrated.enableMultiLinePasteWarning": "never",
    "git.confirmSync": false,
    "git.enableSmartCommit": true,
    "database-client.autoSync": true,
    "mssql.connectionGroups": [
        { "name": "ROOT", "id": "ROOT" }
    ],
    "python.analysis.autoImportCompletions": true,
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.diagnosticSeverityOverrides": {
        "reportUnusedImport": "none" 
    },
    "autoDocstring.docstringFormat": "google",
    "autoDocstring.guessTypes": true,
    "autoDocstring.includeExtendedSummary": true,
    "isort.check": true,
    "isort.importStrategy": "fromEnvironment",
    "python.analysis.importFormat": "absolute",
    "ruff.configuration": {
        "lint": {
            "select": ["I", "RUF022"]
        }
    },
    "ruff.nativeServer": "on",
    "[python]": {
        "editor.codeActionsOnSave": {
            "source.organizeImports": "always",
            "source.fixAll.ruff": "always"
        },
        "editor.defaultFormatter": "astral-sh.ruff",
        "editor.formatOnSave": true,
        "editor.tabSize": 4,
        "editor.insertSpaces": true
    },
    "editor.unicodeHighlight.allowedCharacters": {
        " ": true
    },
    "editor.semanticTokenColorCustomizations": {
        "enabled": true,
        "rules": {
            "keyword": "#569CD6",
            "keyword.control": "#C586C0",
            "keyword.declaration": "#569CD6",
            "storage.type": "#569CD6",
            "builtinConstant": "#569CD6",
            "namespace": "#4EC9B0",
            "module": "#4EC9B0",
            "class.declaration": "#4EC9B0",
            "class": "#4EC9B0",
            "type": "#4EC9B0",
            "typeParameter": "#4EC9B0",
            "stubs": "#4EC9B0",
            "function.declaration": "#DCDCAA",
            "method.declaration": "#DCDCAA",
            "function": "#DCDCAA",
            "method": "#DCDCAA",
            "method.static": "#DCDCAA",
            "method.async": "#DCDCAA", 
            "method.magic": "#DCDCAA",
            "decorator": "#DCDCAA",
            "variable.declaration": "#9CDCFE",
            "variable": "#9CDCFE",
            "parameter": "#9CDCFE",
            "selfParameter": "#9CDCFE",
            "clsParameter": "#9CDCFE",
            "property": "#9CDCFE",
            "variable.readonly": "#4FC1FF",
            "number": "#B5CEA8",
            "string": "#CE9178",
            "comment": "#6A9955",
            "operator": "#D4D4D4",
            "punctuation": "#D4D4D4"
        }
    },
    "editor.tokenColorCustomizations": {
        "textMateRules": [
            {
                "name": "POLICJA SYMBOLI (NIE BLOKUJE GÓRY)",
                "scope": [
                    "punctuation.separator.period.python",
                    "punctuation.separator.element.python",
                    "punctuation.parenthesis.begin.python",
                    "punctuation.parenthesis.end.python",
                    "punctuation.bracket.begin.python",
                    "punctuation.bracket.end.python",
                    "keyword.operator.assignment.python"
                ],
                "settings": {
                    "foreground": "#D4D4D4",
                    "fontStyle": ""
                }
            },
            {
                "name": "Naprawa słowa lambda (Fioletowy)",
                "scope": [
                    "keyword.control.flow.lambda.python",
                    "storage.type.function.lambda.python"
                ],
                "settings": { "foreground": "#C586C0" }
            },
            {
                "name": "Naprawa znaku @ dekoratora (Żółty)",
                "scope": ["punctuation.definition.decorator.python"],
                "settings": { "foreground": "#DCDCAA" }
            },
            {
                "name": "Naprawa dwukropka (Szary)",
                "scope": [
                    "punctuation.section.function.begin.python",
                    "punctuation.section.class.begin.python",
                    "punctuation.separator.dict.python"
                ],
                "settings": { "foreground": "#D4D4D4" }
            },
            {
                "name": "Naprawa tekstów i cudzysłowów (Pomarańczowy)",
                "scope": [
                    "punctuation.definition.string.begin",
                    "punctuation.definition.string.end"
                ],
                "settings": { "foreground": "#CE9178" }
            },
            {
                "name": "Naprawa komentarzy (Zielony)",
                "scope": ["punctuation.definition.comment"],
                "settings": { "foreground": "#6A9955" }
            },
            {
                "name": "WYMUSZENIE KOLORU DEF I CLASS (OSTATECZNE)",
                "scope": [
                    "storage.type.function.python",
                    "storage.type.class.python",
                    "storage.modifier.declaration.python",
                    "keyword.declaration.python"
                ],
                "settings": {
                    "foreground": "#569CD6" 
                }
            }
        ]
    }
}
EOF

echo "✅ Plik settings.json został przygotowany."
echo "🎉 KONFIGURACJA ZAKOŃCZONA!"
