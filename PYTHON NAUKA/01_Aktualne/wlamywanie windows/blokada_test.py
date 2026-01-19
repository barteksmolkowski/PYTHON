import ctypes
import getpass
import os
import subprocess
import sys
import time


class UltimateShield:
    def __init__(self):
        self.user = getpass.getuser()
        sys32 = r"C:\Windows\System32"
        if os.path.exists(sys32):
            self.targets = [f for f in os.listdir(sys32) if f.lower().startswith("wpc") and f.endswith(".exe")]
        else:
            self.targets = ["WpcMon.exe", "WpcTok.exe"]
        self.service = "WpcMonSvc"

    def request_admin(self):
        if not ctypes.windll.shell32.IsUserAnAdmin():
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{os.path.abspath(__file__)}"', None, 1)
            sys.exit()

    def verify_system(self):
        print(f"\n=== RAPORT BEZPIECZEŃSTWA: {time.strftime('%H:%M:%S')} ===")
        print(f"[*] Badany użytkownik: {self.user}")
        print("-" * 45)

        res = subprocess.run(f"net user {self.user}", capture_output=True, shell=True)
        data = res.stdout.decode('cp852', errors='ignore')
        
        if "All" in data or "Wszystkie" in data:
            print(" >> TEST 1 (Czas): [GIT] Dostęp nielimitowany (All).")
        else:
            print(" >> TEST 1 (Czas): [!!!] Wykryto aktywne limity!")

        fw_res = subprocess.run('netsh advfirewall firewall show rule name="X_WpcMon.exe_out"', capture_output=True, shell=True)
        fw_data = fw_res.stdout.decode('cp852', errors='ignore')
        if "X_WpcMon.exe_out" in fw_data:
            print(" >> TEST 2 (Sieć): [GIT] Firewall blokuje Family Link.")
        else:
            print(" >> TEST 2 (Sieć): [!!!] Brak blokady w Firewallu!")

        svc_res = subprocess.run(f"sc query {self.service}", capture_output=True, shell=True)
        svc_data = svc_res.stdout.decode('cp852', errors='ignore')
        if "STOPPED" in svc_data:
            print(" >> TEST 3 (Usługa): [GIT] Mechanizm blokady zatrzymany.")
        else:
            print(" >> TEST 3 (Usługa): [!!!] Usługa nadal aktywna!")
        
        print("-" * 45)

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
            os.system(f"title TARCZA + AUDYTOR - {self.user}")
            self.apply_defense()
            self.verify_system()
            
            choice = input("\nCzy przejść w tryb DUCHA? [t/n]: ").lower()
            if choice == 't':
                print("Uruchamiam w tle...")
                pw = sys.executable.replace("python.exe", "pythonw.exe")
                subprocess.Popen([pw, os.path.abspath(__file__), "ghost"], creationflags=0x08000000)
                sys.exit()

        try:
            while True:
                self.apply_defense()
                if not ghost_mode:
                    print(f"[{time.strftime('%H:%M:%S')}] Tarcza trzyma. Status: OK", end="\r")
                time.sleep(5)
        except KeyboardInterrupt:
            sys.exit()

if __name__ == "__main__":
    shield = UltimateShield()
    shield.request_admin()
    is_ghost = len(sys.argv) > 1 and sys.argv[1] == "ghost"
    shield.monitor(ghost_mode=is_ghost)