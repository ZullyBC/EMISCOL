# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:59:42 2025

@author: Zully JBC

Nota: Este código crea el csv que sera utilizado para entrenar el modelo de RF.
"""

import os
import re
import glob
import xarray as xr
import rasterio
import pandas as pd
import numpy as np

# Función para extraer valor de un raster TIFF en una coordenada (lat, lon)
def extract_raster_value(tif_path, lat, lon):
    with rasterio.open(tif_path) as src:
        try:
            # Verificar si la coordenada está dentro del bbox del raster
            if not (src.bounds.left <= lon <= src.bounds.right and 
                    src.bounds.bottom <= lat <= src.bounds.top):
                return np.nan

            # Obtener índices (row, col) y asegurar que estén dentro del raster
            row, col = src.index(lon, lat)
            if 0 <= row < src.height and 0 <= col < src.width:
                value = src.read(1)[row, col]
            else:
                value = np.nan

        except Exception as e:
            print(f"Error extrayendo valor de {tif_path} en ({lat}, {lon}): {e}")
            value = np.nan
    return value

# Función para extraer valor de una variable en un archivo NetCDF en la posición más cercana
def extract_nc_value(nc_path, lat, lon, var_name):
    try:
        ds = xr.open_dataset(nc_path)
        value = ds[var_name].sel(lat=lat, lon=lon, method="nearest").values.item()
        ds.close()
    except Exception as e:
        print(f"Error extrayendo {var_name} de {nc_path} en ({lat}, {lon}): {e}")
        value = np.nan
    return value

# Función para obtener el archivo anual que contenga el año buscado en el nombre
def get_annual_file(directory, year):
    files = glob.glob(os.path.join(directory, "*.tif"))
    for file in files:
        if str(year) in os.path.basename(file):
            return file
    return None

#Función para abrir archivos de ERA5
# Función para abrir archivos de ERA5 (corregida)
def abrir_dataset_era5(path):
    try:
        ds = xr.open_dataset(path)

        # Renombrar dimensiones 'latitude' y 'longitude' a 'lat' y 'lon'
        if "latitude" in ds.dims:
            ds = ds.rename({"latitude": "lat"})
        if "longitude" in ds.dims:
            ds = ds.rename({"longitude": "lon"})

        # Asegurar que las coordenadas se llamen 'lat' y 'lon'
        if "lat" not in ds.coords:
            ds = ds.assign_coords(lat=("lat", ds["lat"].data))
        if "lon" not in ds.coords:
            ds = ds.assign_coords(lon=("lon", ds["lon"].data))

        # Eliminar dimensiones redundantes si existen
        if "latitude" in ds.dims:
            ds = ds.drop_dims("latitude")
        if "longitude" in ds.dims:
            ds = ds.drop_dims("longitude")

        return ds

    except Exception as e:
        print(f"Error abriendo {path}: {e}")
        return None

# Directorios de tus datos (modificalos según tu estructura)
dir_oco2          = "./OCO2_L2_Lite_FP"         # 995 archivos .nc4
dir_era5          = "./ERA5"                    # 995 archivos .nc
dir_carbon        = "./Carbon_Tracker"          # 995 archivos .nc
dir_strm          = "./SRTM"                    # 1 archivo .tif (ej: "SRTM_Colombia_pos.tif")
dir_modis_ndvi    = "./MODIS/NDVI"      # 6 archivos .tif (por año)
dir_modis_land    = "./MODIS/Land_Cover"   # 6 archivos .tif (por año)
dir_world_pop     = "./World_pop"               # 6 archivos .tif (por año)

# Listas de archivos (ordenados para emparejar)
oco2_files       = sorted(glob.glob(os.path.join(dir_oco2, "*.nc4")))
era5_files       = sorted(glob.glob(os.path.join(dir_era5, "*.nc")))
carbon_files     = sorted(glob.glob(os.path.join(dir_carbon, "*.nc")))

# Archivo SRTM (suponemos que es uno solo)
strm_files = glob.glob(os.path.join(dir_strm, "*.tif"))
if len(strm_files) > 0:
    strm_file = strm_files[0]
else:
    raise FileNotFoundError("No se encontró archivo SRTM!")

data_list = []

# Iteramos sobre cada archivo OCO-2
for i, oco2_path in enumerate(oco2_files):
    try:
        ds_oco2 = xr.open_dataset(oco2_path)
    except Exception as e:
        print(f"Error abriendo {oco2_path}: {e}")
        continue

    # Usamos regex para extraer la fecha del nombre del archivo OCO-2.
    # Ejemplo del nombre: "oco2_LtCO2_140906_B11100Ar_230523232559s_co.nc4"
    match = re.search(r'oco2_LtCO2_(\d{6})', os.path.basename(oco2_path))
    if match:
        date_str = match.group(1)  # "140906"
        try:
            timestamp = pd.to_datetime(date_str, format="%y%m%d")
            year = timestamp.year
        except Exception as e:
            print(f"Error convirtiendo {date_str} a fecha: {e}")
            timestamp = None
            year = None
    else:
        print(f"No se encontró fecha en el nombre del archivo {oco2_path}")
        timestamp = None
        year = None

    # Para extraer datos de xCO2 y las coordenadas: asumimos que las variables se llaman "xco2", "lat" y "lon"
    try:
        # Ahora usamos DataArray.notnull() para que mask sea un DataArray válido
        xco2_da = ds_oco2["xco2"].values
        # Extraemos lat/lon y xco2 sólo donde mask es True
        lats1 = ds_oco2["latitude"].values
        lons1 = ds_oco2["longitude"].values
        
        lats2, lons2 = np.meshgrid(lats1, lons1, indexing="ij")
        mask = ~np.isnan(xco2_da)
        
        lats   = lats2[mask]
        lons  = lons2[mask]
        xco2_vals  = xco2_da[mask]
    except Exception as e:
        print(f"Error procesando variables en {oco2_path}: {e}")
        ds_oco2.close()
        continue

    # Asumimos que los archivos ERA5 y Carbon Tracker están emparejados por el orden en las listas
    try:
        era5_path = era5_files[i]
        carbon_path = carbon_files[i]
    except IndexError:
        print("No hay suficientes archivos en ERA5 o Carbon Tracker para emparejar.")
        ds_oco2.close()
        break

    try:
        ds_era5 = abrir_dataset_era5(era5_path)
        
    except Exception as e:
        print(f"Error abriendo {era5_path}: {e}")
        ds_oco2.close()
        continue
        
    try:
        ds_carbon = xr.open_dataset(carbon_path)
        
    except Exception as e:
        print(f"Error abriendo {carbon_path}: {e}")
        ds_oco2.close()
        ds_era5.close()
        continue

    # Iteramos sobre cada punto con datos válidos de xco2 en el archivo OCO-2
    for lat, lon, xco2 in zip(lats, lons, xco2_vals):
        # Extraemos variables de ERA5 (suponiendo nombres: t2m, d2m, sp, u10 y v10)
        era5_vars = {}
        # Dentro del bucle donde se extraen variables de ERA5:
        for var in ['t2m', 'd2m', 'sp', 'u10', 'v10']:
            try:
                era5_vars[var] = ds_era5[var].sel(lat=lat, lon=lon, method="nearest").values.item()
            except Exception as e:
                print(f"Error extrayendo {var} de {era5_path} en ({lat}, {lon}): {e}")
                era5_vars[var] = np.nan
        # Extraemos variable de Carbon Tracker (ajusta el nombre de la variable si es distinto)
        # Dentro del bucle donde se extrae 'xco2' de Carbon Tracker:
        try:
            co2_carbon = ds_carbon["xco2"].sel(lat=lat, lon=lon, method="nearest").values.item()
        except Exception as e:
            print(f"Error extrayendo 'xco2' de {carbon_path} en ({lat}, {lon}): {e}")
            co2_carbon = np.nan

        # Valor de SRTM (único para todo el dataset)
        srtm_val = extract_raster_value(strm_file, lat, lon)

        # Para los archivos anuales de MODIS y World_pop, se busca el que corresponda al año extraído
        if year is None:
            print(f"No se pudo determinar el año para {oco2_path}, asignando default 2014")
            year_sel = 2014
        else:
            year_sel = year

        ndvi_path = get_annual_file(dir_modis_ndvi, year_sel)
        land_cover_path = get_annual_file(dir_modis_land, year_sel)
        world_pop_path = get_annual_file(dir_world_pop, year_sel)

        # Validamos que se hayan encontrado los archivos
        if ndvi_path is None or land_cover_path is None or world_pop_path is None:
            print(f"Archivos anuales no encontrados para el año {year_sel}. Skipping point.")
            continue

        ndvi_val = extract_raster_value(ndvi_path, lat, lon)
        land_cover_val = extract_raster_value(land_cover_path, lat, lon)
        world_pop_val = extract_raster_value(world_pop_path, lat, lon)

        # Armamos el diccionario para esta fila del DataFrame
        data_list.append({
            "lat": lat,
            "lon": lon,
            "xco2": xco2,
            "t2m": era5_vars.get("t2m", np.nan),
            "d2m": era5_vars.get("d2m", np.nan),
            "sp": era5_vars.get("sp", np.nan),
            "u10": era5_vars.get("u10", np.nan),
            "v10": era5_vars.get("v10", np.nan),
            "co2_carbon": co2_carbon,
            "srtm": srtm_val,
            "ndvi": ndvi_val,
            "land_cover": land_cover_val,
            "world_pop": world_pop_val,
            "timestamp": timestamp
        })

    ds_oco2.close()
    ds_era5.close()
    ds_carbon.close()

# Creamos el DataFrame final
df = pd.DataFrame(data_list)
print("DataFrame armado con éxito. Número de registros:", len(df))

# Guardamos el DataFrame a CSV (opcional)
df.to_csv("data_ml_oco2.csv", index=False)
