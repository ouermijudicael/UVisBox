# Flow datasets
from .flow2d import double_gyre
from .flow3d import flowmap_3d

# Ensemble and uncertainty datasets
from .ens_uv import load_data as load_uv_ensemble
from .darcy_flow_NN import load_data as load_darcy_nn

# Hurricane track data
from .irma2017_perturbed_tracks import load_dataset as load_irma_tracks

# Temperature and climate data
from .sea_surface_temp_data import load_dataset as load_sst_data
from .temperature_and_wind_data import load_dataset as load_era5_data

__all__ = [
    'double_gyre',
    'flowmap_3d',
    'load_uv_ensemble',
    'load_darcy_nn',
    'load_irma_tracks',
    'load_sst_data',
    'load_era5_data'
]