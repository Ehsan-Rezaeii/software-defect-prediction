"""Lint and format-check every Python fenced block with the installed Ruff."""

import re
import subprocess
from pathlib import Path


def main():
    count = 0
    for path in [Path("README.md"), *Path("docs").rglob("*.md")]:
        for index, match in enumerate(
            re.finditer(r"```(?:python|py)\n(.*?)```", path.read_text(), re.S)
        ):
            code = match.group(1)
            label = f"{path.stem}_{index}.py"
            for args in [["check"], ["format", "--check"]]:
                subprocess.run(
                    ["ruff", *args, "--stdin-filename", label, "-"],
                    input=code,
                    text=True,
                    check=True,
                )
            count += 1
    print(f"Checked {count} Python Markdown fences")


if __name__ == "__main__":
    main()
