"""Sphinx configuration for fastcxt documentation."""

project = "fastcxt"
copyright = "2025–2026, Kevin Korfmann"
author = "Kevin Korfmann"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_title = "fastcxt"
html_theme_options = {
    "source_repository": "https://github.com/kevinkorfmann/fastcxt",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "tskit": ("https://tskit.dev/tskit/docs/stable/", None),
}

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

suppress_warnings = ["epub.unknown_project_files"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
