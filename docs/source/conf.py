# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

project = 'UVisBox'
copyright = '2025, Timbwaoga A. J. Ouermi and Jixian Li'
author = 'Timbwaoga A. J. Ouermi and Jixian Li'
release = '0.0.1'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_theme = 'sphinx_rtd_theme' # Specify your theme
html_static_path = ['_static'] # Path to static files
html_logo = '_static/UVisBox-logo.png' # Path to your logo image

html_theme_options = {
'logo_only': True, # Display only the logo without the project name
'display_version': False, # Hide version number
}
extensions = ["sphinx.ext.autodoc", 
              "sphinx.ext.napoleon", 
              "sphinx.ext.viewcode"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

