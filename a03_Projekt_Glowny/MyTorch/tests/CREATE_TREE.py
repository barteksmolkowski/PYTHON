import os
import sys


class TreeGenerator:
    def __init__(self, target_folder="a03_Projekt_Glowny"):
        self.target_folder = target_folder
        self.exclude = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".vscode",
            ".idea",
            "venv",
            ".env",
        }

    def get_level(self, root):
        parts = root.split(os.sep)
        if self.target_folder in parts:
            idx = parts.index(self.target_folder)
            return len(parts[idx:]) - 1
        return -1

    def generate(self, startpath=".", current_file=None):
        tree_lines = [f"STRUCTURE FOR: '{self.target_folder}'", "=" * 35]
        found_any = False

        for root, dirs, files in os.walk(startpath):
            dirs[:] = sorted([d for d in dirs if d not in self.exclude])

            if self.target_folder not in root:
                continue

            found_any = True
            level = self.get_level(root)

            prefix = str(level) if level > 0 else ""

            folder_display = os.path.basename(root) + "/"

            valid_files = []
            for f in sorted(files):
                if f.startswith(".") or f == "STRUCTURE.txt":
                    continue
                marker = " <--- (YOU ARE HERE)" if f == current_file else ""
                valid_files.append(f"{f}{marker}")

            if valid_files:
                tree_lines.append(f"{prefix}{folder_display}: {','.join(valid_files)}")
            else:
                tree_lines.append(f"{prefix}{folder_display}")

        if not found_any:
            tree_lines.append(f"Nie znaleziono: '{self.target_folder}'")

        output = "\n".join(tree_lines)
        print(output)

        with open("STRUCTURE.txt", "w", encoding="utf-8") as f:
            f.write(output)
        return output


if __name__ == "__main__":
    TARGET = "a03_Projekt_Glowny"
    if len(sys.argv) > 1:
        TARGET = sys.argv[1]
    caller = os.path.basename(sys.argv[0])

    gen = TreeGenerator(target_folder=TARGET)
    gen.generate(startpath=".", current_file=caller)
