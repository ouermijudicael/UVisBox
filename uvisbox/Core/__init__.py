"""
UVisBox Core Module

This module contains the core functionality and fundamental algorithms used throughout
the UVisBox package. It includes statistical methods, color operations, interpolation
algorithms, and cell crossing probability calculations.

The Core module is organized into the following submodules:
- BandDepths: Core statistical methods for band depth calculations
- CellCrossingProb: Cell crossing probability algorithms  
- Colors: Core color interpolation and tree data structures
- CommonInterface: Shared interfaces and configurations across modules
- Interpolations: Core interpolation algorithms
"""

# Import all core functionality
from .BandDepths import *
from .CellsCrossingProb import *
from .Colors import *
from .CommonInterface import *
from .Interpolations import *