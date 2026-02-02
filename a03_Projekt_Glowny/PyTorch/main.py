import sys

import pytest

from common_utils import setup_logging
from core import BrainEngine


def main():
    setup_logging()
    engine = BrainEngine()
    
    if len(sys.argv) < 2:
        print("Use: python main.py [test|nauka|zgaduj <path>]")
        return

    mode = sys.argv[1].lower()

    if mode == "test":
        pytest.main(["tests/", "-v", "--log-cli-level=INFO"])
    
    elif mode == "learning":
        engine.run_training(epochs=10)
        
    elif mode == "guess":
        path = sys.argv[2]
        engine.recognize_digit(path)

if __name__ == "__main__":
    main()
