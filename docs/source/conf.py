# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.append(os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QuantNBody'
copyright = '2022, Saad Yalouz'
author = 'Saad Yalouz'
release = '1.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              # 'sphinx_rtd_theme',
              'nbsphinx',
              'sphinxemoji.sphinxemoji',
              'IPython.sphinxext.ipython_console_highlighting']

napoleon_numpy_docstring = True 

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = ["psi4"]  # To prevent errors to be raised because of lack of psi4 on PyPi

html_static_path = ['_static']
html_logo = "_static/logo2.png"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_theme = 'press'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
}
