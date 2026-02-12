import logging

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = RotatingFileHandler("project.log", maxBytes=10**6, backupCount=3)
file_handler.setLevel(logging.DEBUG)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[console_handler, file_handler],
)

x, y = 10, 20
source = "https://api.test.pl"

print(
    "logger = logging.getLogger('nazwa_logging') # gdy nie chcemy automatycznej nazwy"
)
print("logger = logging.getLogger(__name__) # gdy automatyczna nazwa")

logger = logging.getLogger(__name__)

logger.debug("System values: x=%s, y=%s", x, y)

logger.info("ACTION: Data fetching started for source: %s", source)

logger.warning("SKIP: Metadata file missing, using default values.")

logger.error("FAILED: Connection timeout after 3 retries.")

logger.critical("HALT: Database directory is not writable!")

logger.exception("CRITICAL: Failed to parse JSON file")

try:
    wynik = 1 / 0
except ZeroDivisionError:
    logger.exception("nie dzieli sie na zero")

import logging
import os
from logging.handlers import RotatingFileHandler

# 1. FILTRACJA BIBLIOTEK (Często biblioteki jak 'requests' śmiecą w konsoli)
# Ustawiamy, że od requests chcemy widzieć tylko WARNING i wyżej
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# 3. GŁÓWNA KONFIGURACJA (Łączymy to wszystko)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[console_handler, file_handler],  # Dodajemy oba kanały
)

logger = logging.getLogger("AdvancedProject")

img_id = 404
logger.info(f"ACTION: Saving image with ID: {img_id}")

try:
    with open("nieistniejacy_plik.txt", "r") as f:
        content = f.read()
except FileNotFoundError:
    logger.exception("FAILED: File access error")

print("\n[SYSTEM] Check 'project.log' to see all DEBUG logs!")
