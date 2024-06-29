# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess
import sys

import adam

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../src/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "adam"
copyright = "2021, Artificial and Mechanical Intelligence Lab"
author = "Artificial and Mechanical Intelligence Lab"
# get release from git tag

release = (
    subprocess.check_output(["git", "describe", "--tags", "--always"]).decode().strip()
)


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.inheritance_diagram",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx.ext.napoleon",
    "autoapi.extension",
]

autoapi_dirs = ["../src/"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

autosummary_generate = True
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = []

html_title = f"adam"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
# html_theme = "sphinx_rtd_theme"

# html_theme = "furo"
# html_logo = "pirati.png"

html_context = {
    "display_github": True,
    "github_user": "TOB-KNPOB",
    "github_repo": "Jabref2Obsdian",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_static_path = ["_static"]
