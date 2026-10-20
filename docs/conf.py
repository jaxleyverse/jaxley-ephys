"""Sphinx configuration of the jaxley-ephys documentation.

Build locally with ``uv run --group docs sphinx-build -W docs docs/_build/html``.
"""

import jaxley_ephys

project = "jaxley-ephys"
author = "jaxley-ephys contributors"
copyright = "2026, jaxley-ephys contributors"
version = release = jaxley_ephys.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_build", "examples/README.md"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

# The example notebooks (docs/examples links to ../examples) are stored with their
# outputs: running them needs the Allen test data.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

templates_path = ["_templates"]  # classes without a separate __init__ entry
autosummary_generate = True
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_member_order = "bysource"
add_module_names = False

html_title = "jaxley-ephys"
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/jaxleyverse/jaxley-ephys",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_download_button": False,
    "show_toc_level": 2,
}
