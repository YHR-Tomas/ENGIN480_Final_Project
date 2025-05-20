import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import pyproj
import os
import numpy as np
import geojson
from py_wake.examples.data.hornsrev1 import V80
from py_wake import NOJ
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014, Zong_PorteAgel_2020, Niayifar_PorteAgel_2016
from py_wake.site import xrsite
from py_wake.site.shear import PowerShear
UniformWeibullSite = xrsite.UniformWeibullSite

# === Filepaths ===
current_directory = os.path.dirname(os.path.abspath(__file__))
filepaths = [
    ("Revolution SouthFork Wind", 
     os.path.join(current_directory, 'Layouts', 'Revolution_SouthFork_Wind_Boundary.geojson'),
     os.path.join(current_directory, 'Layouts', 'Revolution_SouthFork_Wind_line_offshore.geojson')),
    ("Vineyard Wind",
     os.path.join(current_directory, 'Layouts', 'vineyardwind.geojson'),
     os.path.join(current_directory, 'Layouts', 'vineyardwind_TBL.geojson')),
    ("Coastal Virginia",
     os.path.join(current_directory, 'Layouts', 'coastal_virginia.geojson'),
     os.path.join(current_directory, 'Layouts', 'coastal_virginia_TBL.geojson')),
    ("Fecamp",
     os.path.join(current_directory, 'Layouts', 'Fecamp.geojson'),
     os.path.join(current_directory, 'Layouts', 'Fecamp_TBL.geojson')),
]

def convert_LatLong_to_utm(long, lat):
    wgs84 = pyproj.CRS('EPSG:4326')
    utm_zone = int((long + 180) / 6) + 1
    hemisphere = 'N' if lat >= 0 else 'S'
    utm_epsg_code = f'EPSG:{32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone}'
    utm = pyproj.CRS(utm_epsg_code)
    transformer = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)
    return transformer.transform(long, lat)

def get_lat_long(filepath):
    with open(filepath, 'r') as file:
        geojson_data = geojson.load(file)
    coordinates = geojson_data["features"][0]["geometry"]["coordinates"]
    return coordinates

def plot_bound_on_ax(ax, title, boundary_path, offshore_path):
    boundary_coords = get_lat_long(boundary_path)
    utm_boundary = [convert_LatLong_to_utm(lon, lat) for lon, lat in boundary_coords]
    line = LineString(utm_boundary)
    gdf = gpd.GeoDataFrame({"geometry": [line]})
    gdf.plot(ax=ax, color='black', linewidth=2)

    latlon_coords = get_lat_long(offshore_path)
    wt_x, wt_y = zip(*[convert_LatLong_to_utm(lon, lat) for lon, lat in latlon_coords])
    ax.scatter(wt_x, wt_y, marker='.', c='red', s=10)
    ax.set_title(title)
    ax.set_xlabel("X (UTM)")
    ax.set_ylabel("Y (UTM)")
    ax.grid(True)
    return wt_x, wt_y

fig, axs = plt.subplots(2, 2, figsize=(10, 7))
axs = axs.flatten()
layout_by_farm = {}

for i, (title, boundary_fp, offshore_fp) in enumerate(filepaths):
    x, y = plot_bound_on_ax(axs[i], title, boundary_fp, offshore_fp)
    layout_by_farm[title] = (x, y)

plt.tight_layout()
plt.show()

# === Wind Profile Extraction ===
lib_filenames = {

    "Vineyard Wind": "vineyardwind_gwa3_gwc_customarea.lib",
    "Coastal Virginia": "coastalvirginia_gwa3_gwc_customarea.lib",
    "Revolution SouthFork Wind": "Revolution_SouthFork_Wind_wind_spped.lib",
    "Fecamp": "fecamp_gwa3_gwc_customarea.lib"

}
lib_paths = {name: os.path.join(current_directory, 'gwc', filename) for name, filename in lib_filenames.items()}

def extract_f_a_k(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    f = [float(x) for x in lines[4].strip().split()]
    a = [float(x) for x in lines[13].strip().split()]
    k = [float(x) for x in lines[14].strip().split()]
    return f, a, k

results_by_farm = {}

for name, _, _ in filepaths:
    print(f"\n=== Running AEP simulation for: {name} ===")
    f, a, k = extract_f_a_k(lib_paths[name])
    wt_x, wt_y = layout_by_farm[name]

    class CustomSite(UniformWeibullSite):
        def __init__(self, ti=.1, shear=PowerShear(h_ref=200, alpha=0.1)): 
            UniformWeibullSite.__init__(self, np.array(f)/np.sum(f), a, k, ti=ti, shear=shear)
            self.initial_position = np.array([wt_x, wt_y]).T

    site = CustomSite()
    wind_turbines = V80()

    # Show power and CT curves
    ws_range = np.arange(3, 25, 1)
    plt.figure()
    plt.plot(ws_range, wind_turbines.power(ws_range)/1E6)
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Power [MW]')
    plt.title(f'Power Curve - {name}')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(ws_range, wind_turbines.ct(ws_range))
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('CT [-]')
    plt.title(f'Thrust Coefficient - {name}')
    plt.grid(True)
    plt.show()

    print(site.ds)
    print(site.ds['wd'])
    print(site.ds['ws'])
    print(site.ds['Sector_frequency'])
    print(site.ds['Sector_frequency'].isel(wd=3))

    _ = site.plot_wd_distribution(ws_bins=[0,5,10,15,20,25])
    
print("done")
"""
    sim_noj = NOJ(site, wind_turbines)
    sim_bast = Bastankhah_PorteAgel_2014(site, wind_turbines, k=0.032)
    sim_zong = Zong_PorteAgel_2020(site, wind_turbines)
    sim_niayifar = Niayifar_PorteAgel_2016(site, wind_turbines)

    aep_noj = sim_noj(wt_x, wt_y).aep().sum()
    aep_bast = sim_bast(wt_x, wt_y).aep().sum()
    aep_zong = sim_zong(wt_x, wt_y).aep().sum()
    aep_niayifar = sim_niayifar(wt_x, wt_y).aep().sum()

    results_by_farm[name] = {
        "NOJ": aep_noj,
        "Bastankhah": aep_bast,
        "Zong": aep_zong,
        "Niayifar": aep_niayifar
    }

# === Plot AEP Comparison ===
for farm_name, result in results_by_farm.items():
    models = list(result.keys())
    values = [result[m] for m in models]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, values, color=['black', 'gray', 'dimgray', 'silver'])
    plt.ylabel('AEP [MWh]')
    plt.title(f'AEP Comparison for {farm_name}')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

print("done")
"""