"""Sphinx configuration."""
project = "Zerodose"
author = "Christian Hinge"
copyright = "2023, Christian Hinge"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
