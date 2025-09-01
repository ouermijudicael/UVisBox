import xarray as xr

def load_dataset():
    """ 
    Load the ERA5 EDA 3D wind and temperature dataset.
    """
    ds = xr.open_dataset("era5_eda_3d_wind_temp_us_april2_2024/era5eda_us_pl_uvwt_2024_04_02.nc")
    return ds
