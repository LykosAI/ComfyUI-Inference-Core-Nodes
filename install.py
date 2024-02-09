import sys
import subprocess

from pathlib import Path

PARENT_DIR = Path(__file__).parent


def pip_install(package: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package],
        cwd=PARENT_DIR
    )


def main() -> None:
    pip_install("-e .")


if __name__ == '__main__':
    main()
