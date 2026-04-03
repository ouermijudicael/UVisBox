"""
Unified styling configuration for boxplot visualizations.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BoxplotStyleConfig:
    """
    Configuration for median and outlier visualization in statistical boxplots.
    
    This configuration is shared across CurveBoxplot, ContourBoxplot, and 
    FunctionalBoxplot modules to ensure consistent styling.
    
    Attributes:
    -----------
    percentiles : list of float
        Percentiles for the bands to be plotted (default: [25, 50, 75, 90]).
    percentile_colormap : str or matplotlib.colors.Colormap
        Colormap for percentile band colors. Can be a colormap name (e.g., 'viridis')
        or a matplotlib Colormap object. Percentiles are mapped to [0, 1] range
        (percentile/100) to sample colors from the colormap (default: 'viridis').
    
    show_bands : bool
        Whether to display the percentile bands (default: True).

    show_median : bool
        Whether to display the median curve/contour (default: True).
    median_color : str
        Color for the median curve/contour (default: 'red').
    median_width : float
        Line width for the median curve/contour (default: 3.0).
    median_alpha : float
        Transparency level for the median, 0.0 to 1.0 (default: 1.0).
    
    show_outliers : bool
        Whether to display outlier curves/contours (default: False).
    outliers_color : str
        Color for outlier curves/contours (default: 'gray').
    outliers_width : float
        Line width for outlier curves/contours (default: 1.0).
    outliers_alpha : float
        Transparency level for outliers, 0.0 to 1.0 (default: 0.5).
    
    Examples:
    ---------
    >>> # Use defaults (viridis colormap)
    >>> config = BoxplotStyleConfig()
    
    >>> # Use different colormap
    >>> config = BoxplotStyleConfig(percentile_colormap='plasma')
    
    >>> # Use custom matplotlib colormap
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import LinearSegmentedColormap
    >>> custom_cmap = LinearSegmentedColormap.from_list('custom', ['lightblue', 'darkblue'])
    >>> config = BoxplotStyleConfig(percentile_colormap=custom_cmap)
    
    >>> # Customize median only
    >>> config = BoxplotStyleConfig(median_color='blue', median_width=5)
    
    >>> # Hide median, show custom outliers with hot colormap
    >>> config = BoxplotStyleConfig(
    ...     percentile_colormap='hot',
    ...     show_median=False,
    ...     show_outliers=True,
    ...     outliers_color='orange',
    ...     outliers_alpha=0.8
    ... )
    
    >>> # Completely custom
    >>> config = BoxplotStyleConfig(
    ...     percentiles=[25, 50, 75, 95],
    ...     percentile_colormap='coolwarm',
    ...     show_median=True,
    ...     median_color='darkblue',
    ...     median_width=4,
    ...     median_alpha=0.9,
    ...     show_outliers=True,
    ...     outliers_color='purple',
    ...     outliers_width=2,
    ...     outliers_alpha=0.6
    ... )
    """
    
    # Percentile configuration
    percentiles: list = field(default_factory=lambda: [25, 50, 75, 90])
    percentile_colormap: str = 'viridis'
    
    # Band configuration
    show_bands: bool = True

    # Median configuration
    show_median: bool = True
    median_color: str = 'red'
    median_width: float = 3.0
    median_alpha: float = 1.0
    
    # Outliers configuration
    show_outliers: bool = False
    outliers_color: str = 'gray'
    outliers_width: float = 1.0
    outliers_alpha: float = 0.5
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate percentiles
        if not isinstance(self.percentiles, list) or len(self.percentiles) == 0:
            raise ValueError("percentiles must be a non-empty list")
        
        for p in self.percentiles:
            if not 0 <= p <= 100:
                raise ValueError(f"All percentiles must be between 0 and 100, got {p}")
        
        # Validate percentile_colormap
        if isinstance(self.percentile_colormap, str):
            import matplotlib.pyplot as plt
            try:
                plt.get_cmap(self.percentile_colormap)
            except ValueError:
                raise ValueError(f"Invalid colormap name: '{self.percentile_colormap}'")
        # If it's not a string, assume it's a Colormap object (will fail later if not)
        
        # Validate alpha values
        if not 0.0 <= self.median_alpha <= 1.0:
            raise ValueError(f"median_alpha must be between 0.0 and 1.0, got {self.median_alpha}")
        if not 0.0 <= self.outliers_alpha <= 1.0:
            raise ValueError(f"outliers_alpha must be between 0.0 and 1.0, got {self.outliers_alpha}")
        
        # Validate widths
        if self.median_width <= 0:
            raise ValueError(f"median_width must be positive, got {self.median_width}")
        if self.outliers_width <= 0:
            raise ValueError(f"outliers_width must be positive, got {self.outliers_width}")
    
    def get_percentile_colors(self):
        """
        Get colors for percentile bands by sampling from the colormap.
        
        Percentiles are mapped to [0, 1] range using percentile/100, then
        colors are sampled from the colormap at those positions.
        
        Returns:
        --------
        list : List of RGBA tuples (one per percentile)
        
        Examples:
        ---------
        >>> config = BoxplotStyleConfig(percentiles=[0, 50, 100])
        >>> colors = config.get_percentile_colors()
        >>> # Returns colors from start, middle, and end of viridis colormap
        """
        import matplotlib.pyplot as plt
        
        # Get colormap object
        if isinstance(self.percentile_colormap, str):
            cmap = plt.get_cmap(self.percentile_colormap)
        else:
            cmap = self.percentile_colormap
        
        # Map percentiles to [0, 1] range and sample colors
        normalized = [p / 100.0 for p in self.percentiles]
        return [cmap(n) for n in normalized]
