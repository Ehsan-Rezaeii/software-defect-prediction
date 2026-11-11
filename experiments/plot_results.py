"""Render the final report once from immutable saved model outputs."""

from pathlib import Path

from software_defect_prediction.report import report

if __name__ == "__main__":
    report(Path.cwd(), Path("results/study").resolve())
