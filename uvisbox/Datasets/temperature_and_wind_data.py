import xarray as xr
import os

def load_dataset():
    """ 
    Load the ERA5 EDA 3D wind and temperature dataset.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "era5_eda_3d_temp_rh_pv_geo_uvw_us_apr2_2024/era5_eda_us_pl_temp_rh_pv_geo_uvw_2024_04_02.nc")
    # file_path = os.path.join(current_dir, "era5_eda_3d_wind_temp_us_april2_2024/era5eda_us_pl_uvwt_2024_04_02.nc")
    ds = xr.open_dataset(file_path)
    return ds
