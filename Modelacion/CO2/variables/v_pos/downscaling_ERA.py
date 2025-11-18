# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:45:46 2025

@author: Zully JBC

Nota: Código para hacer downscaling de los datos de ERA5, cuya resolución espacial 
original es 0.5° y se baja a 0.05° por medio de interpolación linear y cubica para 
la variable sp (Surface pressure), debido a que estos metodos presentaron bajos
errores en su interpolación, además de un procesamiento óptimo. 
"""

# 1. Importación de librerias
import os
import sys
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
import datetime

# 2. Parámetros del año y rutas
#anio = 2024 
anio = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.datetime.now().year

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Obtener directorio del script actual

# Rutas ABSOLUTAS
input_dir = os.path.join(SCRIPT_DIR, "..", "v_pre", "ERA5", str(anio))
output_dir = os.path.join(SCRIPT_DIR, "ERA5", str(anio))  # Crea en v_pos/ERA5/

os.makedirs(output_dir, exist_ok=True)

desired_res = 0.05 # Resolución deseada en grados

# Variables a procesar y el método de interpolación
vars_to_process = {
    't2m': 'linear',
    'd2m': 'linear',
    'sp': 'cubic',
    'u10': 'linear',
    'v10': 'linear'
}

archivos_procesados = 0

# 3. Función para procesar cada uno de los archivos de la carpeta
def process_file_rgi(file_path, show_plots=False):
    global archivos_procesados
    base_name = os.path.basename(file_path)
    out_filename = os.path.splitext(base_name)[0] + "_co.nc"
    out_file = os.path.join(output_dir, out_filename)

    if os.path.exists(out_file):
        print(f"⏩ {base_name} ya procesado. Omitiendo.")
        return

    print(f"\nProcesando archivo: {base_name}")
    try:
        ds_in = Dataset(file_path, 'r')
    except Exception as e:
        print(f"Error al abrir {file_path}: {e}")
        return

    try:
        lat_orig = ds_in.variables['latitude'][:]
        lon_orig = ds_in.variables['longitude'][:]
    except Exception as e:
        print(f"Error leyendo coordenadas en {base_name}: {e}")
        ds_in.close()
        return

    reverse_data = False
    if lat_orig[0] > lat_orig[-1]:
        print("Las latitudes están en orden descendente, revirtiendo.")
        lat_orig = lat_orig[::-1]
        reverse_data = True

    lat_new_vals = np.arange(np.min(lat_orig), np.max(lat_orig) + desired_res, desired_res)
    lon_new_vals = np.arange(np.min(lon_orig), np.max(lon_orig) + desired_res, desired_res)
    lon_new, lat_new = np.meshgrid(lon_new_vals, lat_new_vals)

    downscaled = {}
    time_index = 0
    lon_orig_2d, lat_orig_2d = np.meshgrid(lon_orig, lat_orig)

    for var, interp_method in vars_to_process.items():
        print(f"  Procesando variable {var}...")
        try:
            data = ds_in.variables[var][time_index, :, :]
            if reverse_data:
                data = data[::-1, :]
        except Exception as e:
            print(f"    Error leyendo {var}: {e}")
            continue

        try:
            if var == 'sp':
                rbs = RectBivariateSpline(lat_orig, lon_orig, data)
                interpolated = rbs(lat_new_vals, lon_new_vals)
            else:
                interp_func = RegularGridInterpolator((lat_orig, lon_orig), data, method='linear')
                pts_dest = np.column_stack((lat_new.flatten(), lon_new.flatten()))
                interpolated = interp_func(pts_dest).reshape(lat_new.shape)

            downscaled[var] = interpolated
        except Exception as e:
            print(f"    Error en interpolación para {var}: {e}")

    ds_in.close()

    try:
        ds_out = Dataset(out_file, 'w', format='NETCDF4')
        ds_out.createDimension('lat', lat_new.shape[0])
        ds_out.createDimension('lon', lon_new.shape[1])
        lat_var = ds_out.createVariable('latitude', 'f4', ('lat',))
        lon_var = ds_out.createVariable('longitude', 'f4', ('lon',))
        lat_var[:] = lat_new_vals
        lon_var[:] = lon_new_vals

        for var, arr in downscaled.items():
            var_out = ds_out.createVariable(var, 'f4', ('lat', 'lon'))
            var_out[:] = arr

        ds_out.source = file_path
        ds_out.close()
        print(f"  ✅ Guardado: {out_file}")
        archivos_procesados += 1
    except Exception as e:
        print(f"  ❌ Error guardando {out_file}: {e}")

#  Procesar archivos del año seleccionado
if os.path.exists(input_dir):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nc')]
    print(f"\n📂 Archivos encontrados en {input_dir}: {len(files)}")
    for file_path in files:
        process_file_rgi(file_path, show_plots=False)
    print(f"\n✅ Procesamiento completo para {anio}. Archivos procesados: {archivos_procesados}\n")
else:
    print(f"🚫 La carpeta de entrada para el año {anio} no existe: {input_dir}")
