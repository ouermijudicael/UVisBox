"""
Curve_Boxplot Module

This module provides curve-based boxplot functionality for uncertainty visualization.
"""

# Import main function
from .curve_boxplot import curve_boxplot

# Import individual pipeline components
from .curve_boxplot_stats import curve_boxplot_summary_statistics
from .curve_boxplot_mesh import curve_boxplot_mesh
from .curve_boxplot_vis import visualize_curve_boxplot

__all__ = [
    'curve_boxplot',
    'curve_boxplot_summary_statistics',
    'curve_boxplot_mesh',
    'visualize_curve_boxplot'
]


# Stats subdirectory is available for future expansion