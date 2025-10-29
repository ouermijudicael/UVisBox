"""
Unit tests for BoxplotStyleConfig
"""
import pytest
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


class TestBoxplotStyleConfigDefaults:
    """Test default configuration values."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = BoxplotStyleConfig()
        
        # Percentile configuration
        assert config.percentiles == [25, 50, 75, 90]
        assert config.percentile_colormap == 'viridis'
        
        # Median configuration
        assert config.show_median is True
        assert config.median_color == 'red'
        assert config.median_width == 3.0
        assert config.median_alpha == 1.0
        
        # Outliers configuration
        assert config.show_outliers is False
        assert config.outliers_color == 'gray'
        assert config.outliers_width == 1.0
        assert config.outliers_alpha == 0.5


class TestBoxplotStyleConfigValidation:
    """Test validation logic in __post_init__."""
    
    def test_empty_percentiles_raises_error(self):
        """Empty percentiles list should raise ValueError."""
        with pytest.raises(ValueError, match="percentiles must be a non-empty list"):
            BoxplotStyleConfig(percentiles=[])
    
    def test_percentile_out_of_range_low(self):
        """Percentile below 0 should raise ValueError."""
        with pytest.raises(ValueError, match="All percentiles must be between 0 and 100"):
            BoxplotStyleConfig(percentiles=[-1, 50, 90])
    
    def test_percentile_out_of_range_high(self):
        """Percentile above 100 should raise ValueError."""
        with pytest.raises(ValueError, match="All percentiles must be between 0 and 100"):
            BoxplotStyleConfig(percentiles=[25, 50, 101])
    
    def test_invalid_colormap_name(self):
        """Invalid colormap name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid colormap name"):
            BoxplotStyleConfig(percentile_colormap='not_a_real_colormap')
    
    def test_median_alpha_out_of_range_low(self):
        """median_alpha below 0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="median_alpha must be between 0.0 and 1.0"):
            BoxplotStyleConfig(median_alpha=-0.1)
    
    def test_median_alpha_out_of_range_high(self):
        """median_alpha above 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="median_alpha must be between 0.0 and 1.0"):
            BoxplotStyleConfig(median_alpha=1.5)
    
    def test_outliers_alpha_out_of_range_low(self):
        """outliers_alpha below 0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="outliers_alpha must be between 0.0 and 1.0"):
            BoxplotStyleConfig(outliers_alpha=-0.1)
    
    def test_outliers_alpha_out_of_range_high(self):
        """outliers_alpha above 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="outliers_alpha must be between 0.0 and 1.0"):
            BoxplotStyleConfig(outliers_alpha=1.5)
    
    def test_median_width_non_positive(self):
        """Non-positive median_width should raise ValueError."""
        with pytest.raises(ValueError, match="median_width must be positive"):
            BoxplotStyleConfig(median_width=0)
        
        with pytest.raises(ValueError, match="median_width must be positive"):
            BoxplotStyleConfig(median_width=-1.5)
    
    def test_outliers_width_non_positive(self):
        """Non-positive outliers_width should raise ValueError."""
        with pytest.raises(ValueError, match="outliers_width must be positive"):
            BoxplotStyleConfig(outliers_width=0)
        
        with pytest.raises(ValueError, match="outliers_width must be positive"):
            BoxplotStyleConfig(outliers_width=-2.0)


class TestBoxplotStyleConfigCustomization:
    """Test custom configuration scenarios."""
    
    def test_custom_percentiles(self):
        """Test custom percentiles configuration."""
        config = BoxplotStyleConfig(percentiles=[10, 50, 90])
        assert config.percentiles == [10, 50, 90]
    
    def test_custom_colormap(self):
        """Test custom colormap configuration."""
        config = BoxplotStyleConfig(percentile_colormap='plasma')
        assert config.percentile_colormap == 'plasma'
    
    def test_get_percentile_colors(self):
        """Test getting colors from colormap."""
        config = BoxplotStyleConfig(percentiles=[0, 50, 100])
        colors = config.get_percentile_colors()
        assert len(colors) == 3
        # Colors should be RGBA tuples
        assert all(len(c) == 4 for c in colors)
    
    def test_get_percentile_colors_mapping(self):
        """Test that percentiles map correctly to colormap range."""
        config = BoxplotStyleConfig(
            percentiles=[0, 25, 50, 75, 100],
            percentile_colormap='viridis'
        )
        colors = config.get_percentile_colors()
        # Should get 5 colors spanning the colormap
        assert len(colors) == 5
    
    def test_custom_median_styling(self):
        """Test custom median styling."""
        config = BoxplotStyleConfig(
            median_color='blue',
            median_width=5.0,
            median_alpha=0.8
        )
        assert config.median_color == 'blue'
        assert config.median_width == 5.0
        assert config.median_alpha == 0.8
    
    def test_custom_outliers_styling(self):
        """Test custom outliers styling."""
        config = BoxplotStyleConfig(
            show_outliers=True,
            outliers_color='orange',
            outliers_width=2.5,
            outliers_alpha=0.7
        )
        assert config.show_outliers is True
        assert config.outliers_color == 'orange'
        assert config.outliers_width == 2.5
        assert config.outliers_alpha == 0.7
    
    def test_hide_median_show_outliers(self):
        """Test hiding median while showing outliers."""
        config = BoxplotStyleConfig(
            show_median=False,
            show_outliers=True
        )
        assert config.show_median is False
        assert config.show_outliers is True
    
    def test_edge_case_percentiles(self):
        """Test edge case percentile values."""
        config = BoxplotStyleConfig(percentiles=[0, 50, 100])
        assert config.percentiles == [0, 50, 100]
    
    def test_edge_case_alpha_values(self):
        """Test edge case alpha values."""
        config = BoxplotStyleConfig(
            median_alpha=0.0,
            outliers_alpha=1.0
        )
        assert config.median_alpha == 0.0
        assert config.outliers_alpha == 1.0
    
    def test_comprehensive_customization(self):
        """Test fully customized configuration."""
        config = BoxplotStyleConfig(
            percentiles=[25, 50, 75, 95],
            percentile_colormap='plasma',
            show_median=True,
            median_color='darkblue',
            median_width=4,
            median_alpha=0.9,
            show_outliers=True,
            outliers_color='purple',
            outliers_width=2,
            outliers_alpha=0.6
        )
        
        assert config.percentiles == [25, 50, 75, 95]
        assert config.percentile_colormap == 'plasma'
        assert config.show_median is True
        assert config.median_color == 'darkblue'
        assert config.median_width == 4
        assert config.median_alpha == 0.9
        assert config.show_outliers is True
        assert config.outliers_color == 'purple'
        assert config.outliers_width == 2
        assert config.outliers_alpha == 0.6
