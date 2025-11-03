"""
FunctionalBoxplot Module

This module provides functional boxplot functionality for uncertainty visualization.
"""

# Import main function
from .functional_boxplot import functional_boxplot

# Import individual pipeline components
from .functional_boxplot_stats import functional_boxplot_summary_statistics
from .functional_boxplot_mesh import functional_boxplot_mesh
from .functional_boxplot_vis import visualize_functional_boxplot

__all__ = [
    'functional_boxplot',
    'functional_boxplot_summary_statistics',
    'functional_boxplot_mesh',
    'visualize_functional_boxplot'
]
