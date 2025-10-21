# import os
# import cdsapi

# # Initialize CDS API client
# c = cdsapi.Client()

# # ----------------------------
# # Settings
# # ----------------------------
# AREA = [50, -125, 24, -66]  # Entire continental US
# YEAR = "2024"
# MONTH = "04"
# DAY = "02"

# PRESSURE_LEVELS = [
#     "1000","925","850","700","500","300"
# ]

# PARAMS = [
#     "temperature",
#     "u_component_of_wind",
#     "v_component_of_wind",
#     "vertical_velocity"
# ]

# TIMES = [f"{h:02d}:00" for h in range(24)]

# OUT_DIR = "era5_eda_3d_wind_temp_us_april2_2024"
# os.makedirs(OUT_DIR, exist_ok=True)
# target = os.path.join(OUT_DIR, f"era5eda_us_pl_uvwt_{YEAR}_{MONTH}_{DAY}.nc")

# # ----------------------------
# # Download
# # ----------------------------
# c.retrieve(
#     "reanalysis-era5-pressure-levels",
#     {
#         "product_type": "ensemble_members",
#         "variable": PARAMS,
#         "pressure_level": PRESSURE_LEVELS,
#         "year": YEAR,
#         "month": MONTH,
#         "day": DAY,
#         "time": TIMES,
#         "area": AREA,          # [N, W, S, E]
#         "format": "netcdf"
#     },
#     target
# )

# print(f"✅ Finished {target}")


import os
import cdsapi

# Initialize CDS API client
c = cdsapi.Client()

# ----------------------------
# Settings
# ----------------------------
AREA = [50, -125, 24, -66]  # [North, West, South, East] — Continental US
YEAR = "2024"
MONTH = "04"
DAY = "02"

PRESSURE_LEVELS = [
    "1000", "925", "850", "700", "500", "300"
]

# Variables of interest
PARAMS = [
    "temperature",          # K
    "relative_humidity",    # %
    "potential_vorticity",  # K m^2 kg^-1 s^-1
    "geopotential",          # m^2 s^-2 (represents height)
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity"
]

TIMES = [f"{h:02d}:00" for h in range(24)]

# Output directory and filename
OUT_DIR = "era5_eda_3d_temp_rh_pv_geo_uvw_us_apr2_2024"
os.makedirs(OUT_DIR, exist_ok=True)
target = os.path.join(OUT_DIR, f"era5_eda_us_pl_temp_rh_pv_geo_uvw_{YEAR}_{MONTH}_{DAY}.nc")

# ----------------------------
# Download request
# ----------------------------
c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        # "product_type": "reanalysis",  # Deterministic version
        "product_type": "ensemble_members",  # Ensemble version
        "variable": PARAMS,
        "pressure_level": PRESSURE_LEVELS,
        "year": YEAR,
        "month": MONTH,
        "day": DAY,
        "time": TIMES,
        "area": AREA,
        "format": "netcdf",
    },
    target
)

print(f"✅ Finished downloading {target}")