import ctypes
import getpass
import os
import subprocess
import sys
import time

import psutil


class UltimateShield:
    def __init__(self):
        self.user = getpass.getuser()
        sys32 = r"C:\Windows\System32"
        if os.path.exists(sys32): self.targets = [f for f in os.listdir(sys32) if f.lower().startswith("wpc") and f.endswith(".exe")]   
        else: self.targets = ["WpcMon.exe", "WpcTok.exe"]
        self.service = "WpcMonSvc"
    def request_admin(self):
        if not ctypes.windll.shell32.IsUserAnAdmin():
            print("Wywołuję uprawnienia administratora...")
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{os.path.abspath(__file__)}"', None, 1)
            sys.exit()
    def verify_system(self):
        print(f"\n=== WERYFIKACJA STANU SYSTEMU (2026) ===\n[*] Użytkownik: {self.user}")
        for exe in self.targets:
            check_cmd = f'netsh advfirewall firewall show rule name="X_{exe}_out"'
            print(f"[GIT] Blokada sieciowa dla {exe}: AKTYWNA" 
                      if os.system(check_cmd + " >nul 2>&1") == 0
                      else f"[!!!] Blokada dla {exe}: BRAK (Zostanie naprawiona w pętli)")
        
        print(f"[GIT] Usługa {self.service}: ZATRZYMANA"
              if "STOPPED" in os.popen(f"sc query {self.service}").read()
              else f"[!!!] Usługa {self.service}: DZIAŁA (Zostanie ubitą)" + "-" * 40)
    def apply_defense(self):
        no_window = 0x08000000
        subprocess.run(f'net user "{self.user}" /time:all', shell=True, creationflags=no_window, capture_output=True)
        subprocess.run(f"sc config {self.service} start= disabled", shell=True, creationflags=no_window, capture_output=True)
        subprocess.run(f"sc stop {self.service}", shell=True, creationflags=no_window, capture_output=True)

        for exe in self.targets:
            path = rf"C:\Windows\System32\{exe}"
            for d in ["in", "out"]:
                cmd = f'netsh advfirewall firewall add rule name="X_{exe}_{d}" dir={d} action=block program="{path}" enable=yes'
                subprocess.run(cmd, shell=True, creationflags=no_window, capture_output=True)
            
            subprocess.run(f"taskkill /F /IM {exe} /T", shell=True, creationflags=no_window, capture_output=True)
    def monitor(self, ghost_mode=False):
        if not ghost_mode:
            os.system(f"title TARCZA AKTYWNA - {self.user}")
            self.verify_system()
            if input("Czy wszystko się zgadza? Chcesz przejść w tryb DUCHA? [t/n]: ") == 't':
                print("Przechodzę w tło... Ochrona będzie działać po cichu.")
                time.sleep(2)
                pw = sys.executable.replace("python.exe", "pythonw.exe")
                subprocess.Popen([pw, os.path.abspath(__file__), "ghost"], creationflags=0x08000000)
                sys.exit()
            print("\nCzuwam w trybie widocznym... Nie zamykaj tego okna!")
        try:
            while True:
                self.apply_defense()
                if not ghost_mode: print(f"[{time.strftime('%H:%M:%S')}] Status: Ochrona trzyma (GIT)", end="\r")
                time.sleep(5)
        except KeyboardInterrupt:
            sys.exit()
if __name__ == "__main__":
    shield = UltimateShield().request_admin().monitor(ghost_mode=len(sys.argv) > 1 and sys.argv[1] == "ghost")
    