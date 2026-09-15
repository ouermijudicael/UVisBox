"""Sphinx configuration for the UVisBox documentation."""

from importlib.metadata import PackageNotFoundError, version as distribution_version
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

# Keep plotting libraries headless if an API module imports them during autodoc.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("SPHINX_BUILD", "1")

project = "UVisBox"
copyright = "2025, Timbwaoga A. J. Ouermi and Jixian Li"
author = "Timbwaoga A. J. Ouermi and Jixian Li"

try:
    release = distribution_version("uvisbox")
except PackageNotFoundError:
    # The source tree remains buildable before the editable package is installed.
    release = "unknown"
version = release

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autoapi_type = "python"
autoapi_dirs = [str(REPOSITORY_ROOT / "uvisbox")]
autoapi_root = "_generated/api"
autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
]
autoapi_ignore = [
    str(REPOSITORY_ROOT / "uvisbox" / "Datasets" / "*" / "*.py"),
    str(REPOSITORY_ROOT / "uvisbox" / "Modules" / "ProbabilisticMarchingSquares" / "plot.py"),
]
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True

templates_path = ["_templates"] if (Path(__file__).parent / "_templates").is_dir() else []
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/UVisBox-logo.png"
html_theme_options = {"logo_only": True}
