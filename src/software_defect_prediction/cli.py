"""Small command-line interface."""

import argparse
from pathlib import Path

from .data import acquire


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "audit-data"])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("results/study"))
    args = parser.parse_args()
    if args.command == "download":
        print(acquire(args.raw, args.output))
    else:
        from .data import EXPECTED, metadata, verified

        for version in EXPECTED:
            print(metadata(verified((args.raw / f"camel-{version}.csv").read_bytes(), version)[0]))
