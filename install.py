import sys
import subprocess

from pathlib import Path

PARENT_DIR = Path(__file__).parent


def pip_install(package: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package], cwd=PARENT_DIR
    )


def try_get_cuda_version() -> str | None:
    try:
        import torch
        return torch.version.cuda
    except ImportError or AttributeError:
        return None


def main() -> None:

    cuda_version = try_get_cuda_version()
    if cuda_version is not None:
        if cuda_version.startswith("12."):
            pip_install("-e .[cuda-12]")
        else:
            pip_install("-e .[cuda]")
    else:
        # Default install
        pip_install("-e .[cpu]")


if __name__ == "__main__":
    main()
