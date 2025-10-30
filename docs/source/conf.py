# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
from unittest.mock import MagicMock

# Set up comprehensive mocking before any imports
class MockModule(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

class MockPyVista(MagicMock):
    class CellType:
        TETRA = 10
        TRIANGLE = 5
        QUAD = 9
        LINE = 3
        VERTEX = 1
        
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Mock heavy dependencies with specific PyVista handling
sys.modules['pyvista'] = MockPyVista()
sys.modules['vtk'] = MockModule()
sys.modules['basemap'] = MockModule()
sys.modules['mpl_toolkits.basemap'] = MockModule()
sys.modules['mayavi'] = MockModule()
sys.modules['mayavi.mlab'] = MockModule()
sys.modules['flask'] = MockModule()
sys.modules['flask.config'] = MockModule()

# sys.path.insert(0, os.path.abspath('../..'))  # Adjust the path as needed
sys.path.insert(0, os.path.abspath('../../uvisbox'))  # Adjust the path as needed
sys.path.insert(0, os.path.abspath('../../examples'))  # Add examples to path

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
}
extensions = ["sphinx.ext.autodoc", 
              "sphinx.ext.napoleon", 
              "sphinx.ext.viewcode"]

# Mock imports for packages that cause issues in CI
autodoc_mock_imports = [
    'pyvista',
    'basemap', 
    'mpl_toolkits.basemap',
    'vtk',
    'vtkmodules',
    'mayavi',
    'mayavi.mlab',
    'flask',
    'flask.config'
]

# Set environment variable to indicate we're building docs
os.environ['SPHINX_BUILD'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

# Configure matplotlib for headless operation
import matplotlib
matplotlib.use('Agg')

# Configure autodoc to skip execution of module-level code
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}

# Suppress some common warnings for cleaner build output
suppress_warnings = [
    'autodoc.import_object',
    'toc.not_readable',
]

# Configure to be less strict about docstring formatting
nitpicky = False

napoleon_google_docstring = True
napoleon_numpy_docstring = True
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

