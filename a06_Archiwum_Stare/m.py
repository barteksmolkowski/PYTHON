import os
import shutil
from pathlib import Path


def wypakuj_i_napraw():
    root = Path(os.getcwd())
    
    print("--- ROZPOCZYNAM WYPAKOWYWANIE PLIKÓW (2026) ---")

    # Przechodzimy przez główne foldery 00, 01, 02...
    for glowny_folder in root.glob("[0-9][0-9]_*"):
        print(f"\nSprawdzam: {glowny_folder.name}")
        
        # Szukamy wewnątrz nich folderów zaczynających się na 'a' i numer
        for podfolder in glowny_folder.glob("a[0-9][0-9]_*"):
            print(f"  |-- Wypakowuję zawartość: {podfolder.name}")
            
            # Przenosimy każdy plik i folder z 'aXX_...' poziom wyżej
            for element in podfolder.iterdir():
                cel = glowny_folder / element.name
                try:
                    if not cel.exists():
                        shutil.move(str(element), str(glowny_folder))
                    else:
                        # Jeśli plik już istnieje, dodajemy prefix, żeby nie nadpisać
                        shutil.move(str(element), str(glowny_folder / f"copy_{element.name}"))
                except Exception as e:
                    print(f"  |-- [BŁĄD] {element.name}: {e}")
            
            # Usuwamy teraz pusty podfolder 'aXX_...'
            try:
                podfolder.rmdir()
                print(f"  |-- [OK] Usunięto pusty: {podfolder.name}")
            except:
                print(f"  |-- [!] Nie usunięto {podfolder.name} (może nie jest pusty)")

    # DODATKOWE PORZĄDKI:
    # 1. Przeniesienie luźnego m.py do Archiwum
    m_py = root / "m.py"
    if m_py.exists():
        shutil.move(str(m_py), str(root / "06_Archiwum_Stare" / "m.py"))
        print("\n[OK] Przeniesiono m.py do 06_Archiwum_Stare")

    print("\n--- KONIEC. Twoja struktura jest teraz płaska (Flat Structure) ---")

if __name__ == "__main__":
    wypakuj_i_napraw()
