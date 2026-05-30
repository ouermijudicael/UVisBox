import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def map_setup(
    data,
    *,
    projection=ccrs.PlateCarree(),
    extent_crs=ccrs.PlateCarree(),
    figsize=(10, 8),
    padding=5.0,
    use_nan_bounds=True,
    coastlines=True,
    coastline_resolution="10m",
    coastline_linewidth=0.5,
    borders=True,
    borders_linewidth=0.5,
    land=True,
    land_facecolor="lightgray",
    land_alpha=0.3,
    ocean=True,
    ocean_facecolor="lightblue",
    ocean_alpha=0.3,
    gridlines=True,
    draw_labels=True,
    gridline_linewidth=0.5,
    gridline_color="gray",
    gridline_alpha=0.5,
    label_size=10,
    show_top_labels=False,
    show_right_labels=False,
    figure_facecolor=None,
):
    """
    Create a Cartopy map figure and axis from lon/lat data.

    Parameters:
    -----------
    data : array-like
        Array containing longitude and latitude in the last dimension.
        Supported shapes include (N, 2) and (..., 2).
        Longitude must be in data[..., 0], latitude in data[..., 1].
    projection : cartopy.crs.Projection, optional
        Map projection used for the axes.
    extent_crs : cartopy.crs.CRS, optional
        CRS used when setting the map extent. Usually PlateCarree for lon/lat data.
    figsize : tuple, optional
        Figure size in inches.
    padding : float, optional
        Padding added to all sides of the computed extent, in degrees.
    use_nan_bounds : bool, optional
        If True, ignore NaN values when computing bounds.
    coastlines, borders, land, ocean : bool, optional
        Toggle map features on or off.
    coastline_resolution : str, optional
        Cartopy coastline resolution, e.g. "110m", "50m", "10m".
    coastline_linewidth, borders_linewidth : float, optional
        Line widths for coastline and border features.
    land_facecolor, ocean_facecolor : str, optional
        Fill colors for land and ocean.
    land_alpha, ocean_alpha : float, optional
        Transparency for land and ocean fills.
    gridlines : bool, optional
        Add gridlines to the map.
    draw_labels : bool, optional
        Show latitude/longitude labels on gridlines.
    gridline_linewidth, gridline_color, gridline_alpha : optional
        Gridline styling.
    label_size : int, optional
        Font size for gridline labels.
    show_top_labels, show_right_labels : bool, optional
        Control label placement for gridlines.
    figure_facecolor : str or None, optional
        Background color for the figure.

    Returns:
    --------
    fig : matplotlib.figure.Figure
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
    """
    arr = np.asarray(data)

    if arr.ndim < 2 or arr.shape[-1] < 2:
        raise ValueError(
            "data must have longitude and latitude in the last dimension, "
            "for example shape (N, 2) or (M, N, 2)."
        )

    lon = arr[..., 0]
    lat = arr[..., 1]

    if use_nan_bounds:
        lon_min = np.nanmin(lon)
        lon_max = np.nanmax(lon)
        lat_min = np.nanmin(lat)
        lat_max = np.nanmax(lat)
    else:
        lon_min = np.min(lon)
        lon_max = np.max(lon)
        lat_min = np.min(lat)
        lat_max = np.max(lat)

    if not np.isfinite([lon_min, lon_max, lat_min, lat_max]).all():
        raise ValueError("Could not compute finite map bounds from data.")

    fig = plt.figure(figsize=figsize, facecolor=figure_facecolor)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=projection)

    ax.set_extent(
        [lon_min - padding, lon_max + padding, lat_min - padding, lat_max + padding],
        crs=extent_crs,
    )

    if coastlines:
        ax.coastlines(resolution=coastline_resolution, linewidth=coastline_linewidth)

    if borders:
        ax.add_feature(cfeature.BORDERS, linewidth=borders_linewidth, edgecolor="black")

    if land:
        ax.add_feature(
            cfeature.LAND,
            facecolor=land_facecolor,
            alpha=land_alpha,
        )

    if ocean:
        ax.add_feature(
            cfeature.OCEAN,
            facecolor=ocean_facecolor,
            alpha=ocean_alpha,
        )

    if gridlines:
        gl = ax.gridlines(
            draw_labels=draw_labels,
            linewidth=gridline_linewidth,
            color=gridline_color,
            alpha=gridline_alpha,
        )
        if draw_labels:
            gl.top_labels = show_top_labels
            gl.right_labels = show_right_labels
            gl.xlabel_style = {"size": label_size}
            gl.ylabel_style = {"size": label_size}

    return fig, ax