import os
import shutil

# Ścieżka do folderu Geometry Dash
# Zmień, jeśli masz inną lokalizację
gd_path = r"C:\Program Files (x86)\Steam\steamapps\common\Geometry Dash"
geode_dll = os.path.join(gd_path, "Geode.dll")
backup_dll = os.path.join(gd_path, "Geode.dll.disabled")

def disable_geode():
    if os.path.exists(geode_dll):
        # zamiast usuwać można zmienić nazwę
        shutil.move(geode_dll, backup_dll)
        print("✅ Geode.dll wyłączony (zmieniono nazwę).")
    else:
        print("⚠️ Nie znaleziono Geode.dll (może już wyłączony).")

def enable_geode():
    if os.path.exists(backup_dll):
        shutil.move(backup_dll, geode_dll)
        print("✅ Geode.dll przywrócony.")
    else:
        print("⚠️ Nie znaleziono Geode.dll.disabled (nie można przywrócić).")

def uninstall_geode():
    if os.path.exists(geode_dll):
        os.remove(geode_dll)
        print("🗑️ Geode.dll usunięty.")
    elif os.path.exists(backup_dll):
        os.remove(backup_dll)
        print("🗑️ Geode.dll.disabled usunięty.")
    else:
        print("⚠️ Geode.dll nie znaleziono (już odinstalowany).")

if __name__ == "__main__":
    print("1. Wyłącz Geode (zmiana nazwy)")
    print("2. Włącz Geode (przywróć)")
    print("3. Odinstaluj Geode (usuń plik)")
    choice = input("Wybierz opcję (1/2/3): ")

    if choice == "1":
        disable_geode()
    elif choice == "2":
        enable_geode()
    elif choice == "3":
        uninstall_geode()
    else:
        print("❌ Nieprawidłowy wybór.")
