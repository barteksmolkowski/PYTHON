import os
import subprocess
import time


def symuluj_atak():
    print("=" * 50)
    print(" ROZPOCZYNAM SYMULACJĘ ATAKU SCREEN TIME ")
    print("=" * 50)

    # 1. Symulacja odpalenia czarnego ekranu (RemoteManagementUI)
    # Tworzymy proces-atrapę, który nazywa się tak samo jak blokada Apple
    print("[*] TEST 1: Próba wyświetlenia blokady ekranu...")
    # Uruchamiamy proces w tle, który nic nie robi, ale ma 'podejrzaną' nazwę
    dummy_ui = subprocess.Popen(
        ["sleep", "100"],
        executable="sleep",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Zmieniamy mu nazwę w systemie na taką, którą śledzi Twoja tarcza
    # (W teście użyjemy nazwy, którą tarcza wyłapuje: RemoteManagementUI)

    time.sleep(2)  # Czekamy na reakcję MacGuarda

    # Sprawdzamy stan procesu
    try:
        pid = (
            subprocess.check_output(["pgrep", "-f", "RemoteManagementUI"])
            .decode()
            .strip()
        )
        state = (
            subprocess.check_output(["ps", "-o", "state=", "-p", pid]).decode().strip()
        )
        if "T" in state:
            print(
                "[SUCCESS] Tarcza wykryła 'RemoteManagementUI' i go ZAMROZIŁA (Status T)."
            )
        else:
            print("[FAIL] 'RemoteManagementUI' nadal działa! Tarcza go nie zatrzymała.")
    except:
        print("[?] Nie znaleziono procesu testowego. Upewnij się, że MacGuard działa.")

    # 2. Test zamrażania licznika czasu
    print("\n[*] TEST 2: Sprawdzanie statusu licznika (parentalcontrolsd)...")
    try:
        pid_t = subprocess.check_output(["pgrep", "parentalcontrolsd"]).decode().strip()
        state_t = (
            subprocess.check_output(["ps", "-o", "state=", "-p", pid_t])
            .decode()
            .strip()
        )
        if "T" in state_t:
            print("[SUCCESS] Licznik czasu jest aktualnie ZAMROŻONY przez tarczę.")
        else:
            print(
                "[INFO] Licznik czasu jest AKTYWNY (Pamiętaj: MacGuard odmraża go na 5s co minutę)."
            )
    except:
        print("[-] Nie znaleziono procesu parentalcontrolsd.")

    # 3. Test szczelności sieci
    print("\n[*] TEST 3: Próba 'oszukania' blokady sieciowej...")
    res = subprocess.run(
        ["ping", "-c", "1", "-t", "1", "://icloud.com"], capture_output=True, text=True
    )
    if "127.0.0.1" in res.stdout:
        print("[SUCCESS] Blokada sieciowa jest nienaruszona.")
    else:
        print("[FAIL] Wykryto wyciek danych do Apple!")

    print("=" * 50)
    print(" KONIEC TESTU BOJOWEGO ")


if __name__ == "__main__":
    symuluj_atak()
