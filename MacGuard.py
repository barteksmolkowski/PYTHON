import os
import subprocess
import time


class MacGuard:
    def __init__(self):
        self.domains = [
            "://apple.com",
            "://apple.com",
            "://apple.com",
            "://apple.com",
            "://icloud.com",
            "://icloud.com",
            "://icloud.com",
            "://apple-cloudkit.com",
        ]
        self.tag = "# --- Apple Support Service ---"
        self.block_proc = "RemoteManagementUI"
        self.timer_proc = "parentalcontrolsd"

    def modify_network(self, block=True):
        """Edytuje plik hosts, aby odciąć raporty do taty"""
        try:
            with open("/etc/hosts", "r") as f:
                lines = f.readlines()

            # Usuwamy stare wpisy z naszym tagiem
            new_lines = [l for l in lines if self.tag not in l]

            if block:
                new_lines.append(f"\n{self.tag}\n")
                for d in self.domains:
                    new_lines.append(f"127.0.0.1 {d} {self.tag}\n")

            with open("/etc/hosts", "w") as f:
                f.writelines(new_lines)

            os.system("killall -HUP mDNSResponder")
            return True
        except Exception as e:
            with open("/tmp/guard.err", "a") as err_log:
                err_log.write(f"Błąd sieci: {e}\n")
            return False

    def run_shield(self):
        """Pętla pilnująca blokady ekranu i licznika czasu"""
        while True:
            try:
                # 1. Neutralizacja czarnego ekranu
                check_ui = subprocess.run(
                    ["pgrep", "-f", self.block_proc], capture_output=True, text=True
                )
                if check_ui.stdout:
                    pid = check_ui.stdout.strip().split("\n")[0]
                    os.system(f"kill -STOP {pid}")
                    os.system(
                        f'osascript -e \'tell application "System Events" to set visible of process "{self.block_proc}" to false\''
                    )

                # 2. Zamrażanie licznika (Ratio: 5s pracy / 55s mrożenia)
                check_timer = subprocess.run(
                    ["pgrep", "-f", self.timer_proc], capture_output=True, text=True
                )
                if check_timer.stdout:
                    t_pid = check_timer.stdout.strip().split("\n")[0]
                    os.system(f"kill -STOP {t_pid}")
                    time.sleep(55)
                    os.system(f"kill -CONT {t_pid}")
                    time.sleep(5)
                else:
                    time.sleep(1)
            except Exception as e:
                time.sleep(5)


if __name__ == "__main__":
    guard = MacGuard()
    # Tryb automatyczny dla LaunchDaemon
    guard.modify_network(block=True)
    guard.run_shield()
