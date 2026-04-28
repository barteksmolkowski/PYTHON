import os
import re
import subprocess

NOT_READY_YET = [
    "ReLULayer",
    "SigmoidLayer",
    "Conv2DLayer",
    "DropoutLayer",
    "FlattenLayer",
    "LinearLayer",
    "Sequential",
    "NeuralNetwork",
    "validate_shape",
    "process_batch",
    "CacheManager",
    "Dataset",
    "DataProcessor",
    "ProjectManager",
    "Sobel",
    "Prewitt",
    "FeatureExtraction",
    "HOG",
    "MSE",
    "CrossEntropy",
    "SGD",
    "Adam",
    "Trainer",
]


def run_filtered_tests():
    start_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(start_dir)

    print(f"Deep Scanning: {start_dir}")

    all_tests = []

    for root, _, files in os.walk(start_dir):
        for f in files:
            if f.lower().startswith("test_") and f.lower().endswith(".py"):
                if f != "AUTO_TESTS.py":
                    all_tests.append(os.path.join(root, f))

    if not all_tests:
        print(f"No test files found in any subdirectories of {start_dir}")
        return

    ready_files = []

    for path in all_tests:
        relative_path = os.path.relpath(path, start_dir)
        is_file_blocked = False
        blocked_reasons = []

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line_clean = line.strip().lower()

                if line_clean.startswith(("import ", "from ")):
                    for m in NOT_READY_YET:
                        m_low = m.lower()
                        pattern = r"\b" + re.escape(m_low) + r"\b"
                        if re.search(pattern, line_clean):
                            blocked_reasons.append(
                                f"Line {line_no}: {line.strip()} (matches '{m}')"
                            )
                            is_file_blocked = True

                if line_clean and not line_clean.startswith(
                    ("import", "from", "#", "@")
                ):
                    break

        if not is_file_blocked:
            ready_files.append(path)
        else:
            print(f"Skipping {relative_path}:")
            for reason in blocked_reasons:
                print(f"    - {reason}")

    if not ready_files:
        print("\nALL tests are currently blocked by unfinished modules.")
        return

    print(f"\nLaunching {len(ready_files)} ready tests from subdirectories...")
    print("─" * 70)

    os.chdir(project_root)
    subprocess.run(["pytest", "-s", "-v"] + ready_files)


if __name__ == "__main__":
    run_filtered_tests()
