"""Sphinx configuration for the pyCurves user and API documentation."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "pyCurves"
author = "pyCurves contributors"
copyright = "2026, pyCurves contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_default_options = {
    "member-order": "bysource",
}
autodoc_typehints = "signature"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
html_theme_options = {
    "description": "DNA and RNA helical analysis in Python",
    "github_user": "JinsenLi",
    "github_repo": "pyCurves",
    "github_button": True,
    "fixed_sidebar": True,
}
html_title = "pyCurves documentation"
html_show_sourcelink = True
