"""Small command-line interface with separate scientific stages."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "download",
            "audit-data",
            "develop",
            "freeze",
            "final",
            "audit-results",
            "report",
            "smoke",
        ],
    )
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("results/study"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if args.command == "download":
        from .data import acquire

        print(acquire(args.raw, output))
    elif args.command == "audit-data":
        from .data import EXPECTED, metadata, verified

        for version in EXPECTED:
            print(metadata(verified((args.raw / f"camel-{version}.csv").read_bytes(), version)[0]))
    elif args.command == "develop":
        from .study import develop

        result = develop(args.raw, output)
        print({k: result[k] for k in ["objective_calls", "objective_fits", "candidate"]})
    elif args.command == "freeze":
        from .study import freeze

        print(freeze(root, output)["base_commit"])
    elif args.command == "final":
        from .study import final

        print(final(args.raw, root, output))
    elif args.command == "audit-results":
        from .audit import audit_results

        print(audit_results(root, output))
    elif args.command == "report":
        from .report import report

        print(report(root, output))
    else:
        from .smoke import smoke

        print(smoke())
