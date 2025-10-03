"""
UVisBox - Uncertainty Visualization Toolbox

A comprehensive Python package for uncertainty visualization, statistical analysis,
and probabilistic visualization methods.

The package is organized into three main components:
- Core: Fundamental algorithms and statistical methods
- Modules: Specialized visualization and analysis modules
- Datasets: Sample datasets and data loading utilities
"""

# Import core functionality
from .Core import *

# Import specialized modules
try:
    from .Modules import *
except ImportError:
    # Modules import might fail during restructuring
    pass

# Import datasets
try:
    from .Datasets import *
except ImportError:
    # Datasets import might fail during restructuring
    pass

# Version information
__version__ = "0.1.0-restructuring"
__author__ = "UVisBox Development Team"