"""Generate API stubs and build the UVisBox documentation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


DOCS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = DOCS_DIR.parent
SOURCE_DIR = DOCS_DIR / "source"
GENERATED_API_DIR = SOURCE_DIR / "_generated" / "api"
BUILD_DIR = DOCS_DIR / "build"


def _remove_generated_directory(path: Path) -> None:
    resolved_path = path.resolve()
    resolved_source = SOURCE_DIR.resolve()
    if resolved_source not in resolved_path.parents:
        raise RuntimeError(
            f"Refusing to remove a path outside {resolved_source}: {resolved_path}"
        )
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def prepare_generated_sources() -> None:
    """Remove API pages left by a previous Sphinx AutoAPI build."""
    _remove_generated_directory(GENERATED_API_DIR)


def build(builder: str, strict: bool) -> None:
    """Build one Sphinx output format from a clean destination."""
    destination = BUILD_DIR / builder
    warning_log = BUILD_DIR / f"{builder}-warnings.log"
    if destination.exists():
        shutil.rmtree(destination)
    warning_log.parent.mkdir(parents=True, exist_ok=True)
    if warning_log.exists():
        warning_log.unlink()

    command = [sys.executable, "-m", "sphinx", "-w", str(warning_log)]
    if strict:
        command.extend(["-W", "--keep-going"])
    command.extend(["-b", builder, str(SOURCE_DIR), str(destination)])

    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("PYVISTA_OFF_SCREEN", "true")
    environment.setdefault("SPHINX_BUILD", "1")
    subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder", default="html", help="Sphinx builder (default: html)")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Allow Sphinx warnings instead of treating them as errors",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_generated_sources()
    build(args.builder, strict=not args.no_strict)


if __name__ == "__main__":
    main()
