"""
UVisBox Modules

This package contains specialized visualization and analysis modules for the UVisBox toolkit.
Each module is organized into Mesh/, Stats/, and Vis/ subdirectories for consistent structure.

Available Modules:
- ContourBoxplot: Contour-based boxplot functionality
- Curve_Boxplot: Curve-based boxplot functionality  
- FunctionalBoxplot: Functional boxplot functionality
- ProbabilisticMarchingCubes: Probabilistic marching cubes algorithms
- ProbabilisticMarchingSquares: Probabilistic marching squares algorithms
- ProbabiliticMarchingTetrahedra: Probabilistic marching tetrahedra algorithms
- ProbabiliticMarchingTriangles: Probabilistic marching triangles algorithms
- SquidGlyph: Squid glyph visualization functionality
- UncertaintyLobes: Uncertainty lobe visualization functionality
- UncertaintyTube: Uncertainty tube functionality
"""

# Import all specialized modules

from .ContourBoxplot import *
from .CurveBoxplot import *
from .FunctionalBoxplot import *
from .ProbabilisticMarchingCubes import *
from .ProbabilisticMarchingSquares import *
from .ProbabilisticMarchingTetrahedra import *
from .ProbabilisticMarchingTriangles import *
from .SquidGlyphs2D import *
from .SquidGlyphs3D import *
from .UncertaintyLobes import *
from .UncertaintyTube import *