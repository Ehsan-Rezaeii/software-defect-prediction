"""Download only the pre-registered immutable Camel releases."""

from pathlib import Path

from software_defect_prediction.data import acquire

if __name__ == "__main__":
    acquire(Path("data/raw"), Path("results/study"))
