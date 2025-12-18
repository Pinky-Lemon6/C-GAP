from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    """Allow running without installation by adding ./src to sys.path."""

    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"

    # Ensure local imports win over any globally-installed cgap
    sys.path.insert(0, str(src_path))

    # Make relative file paths behave as expected when launched elsewhere
    os.chdir(str(repo_root))


def main() -> None:
    _bootstrap_src_path()

    from cgap.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
