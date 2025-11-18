# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:27:24 2025

@author: Zully JBC

Nota: Código para hacer downscaling de los datos de Carbon Tracker, cuya resolución 
espacial original es 3°x2°, por lo que se baja a 0.05° por medio de interpolación
kriging. 
"""

# 1. Importación de librerias
import os
import sys
import numpy as np
from netCDF4 import Dataset
# import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import datetime

# 2. Definición de parámetros de recorte
BBOX = {
    "min_lon": -79.01,
    "max_lon": -66.857,
    "min_lat": -4.226,
    "max_lat": 12.46
}

resolucion = 0.05 # Resolución deseada en grados

# 3. Función para interpolar con kriging.
def kriging_interpolation(x_src, lat_src, lon_src, lon_new_vals, lat_new_vals, variogram_model='linear'):
    """
    Realiza interpolación por kriging (usando Ordinary Kriging) para regrillar los datos.
    
    Parámetros:
      - x_src: valores de xco2 de los puntos recortados (1D).
      - lat_src, lon_src: coordenadas (1D) de los puntos recortados.
      - lon_new_vals, lat_new_vals: arrays 1D para la nueva grilla (creados con np.arange).
      - variogram_model: modelo del variograma ('linear', 'exponential', etc.).
    
    Devuelve:
      - interp_data: una matriz 2D interpolada con dimensiones (len(lat_new_vals), len(lon_new_vals)).
    """
    # Nota: pykrige usa x = lon y y = lat
    OK = OrdinaryKriging(lon_src, lat_src, x_src, variogram_model=variogram_model,
                           verbose=False, enable_plotting=False)
    # OK.execute espera 'grid' y arrays 1D para los puntos de la nueva grilla.
    z, ss = OK.execute('grid', lon_new_vals, lat_new_vals)
    return z  # z tiene forma (len(lat_new_vals), len(lon_new_vals))

# 4. Función para procesar todos los archivos de la carpeta.
def procesar_archivo_ct(archivo_in, archivo_out, bbox, resol, variogram_model):
    """
    Procesa un archivo NetCDF de Carbon Tracker:
      - Abre el archivo y toma la única capa temporal de 'xco2'
      - Usa 'latitude' y 'longitude' para crear la grilla original,
      - Recorta la región de interés según el bounding box,
      - Interpola la variable xco2 a una grilla regular de resolución 'resol' usando kriging,
      - Guarda el resultado en un nuevo archivo NetCDF y muestra un plot para revisión (opcional).
    """
    try:
        with Dataset(archivo_in, 'r') as ds_in:
            # Seleccionar la única capa temporal de xco2 (forma: (lat, lon))
            xco2 = ds_in.variables['xco2'][0, :, :]
            lat = ds_in.variables['latitude'][:]
            lon = ds_in.variables['longitude'][:]
            
            # Crear grilla 2D a partir de lat y lon si son 1D
            if lat.ndim == 1 and lon.ndim == 1:
                lon2d, lat2d = np.meshgrid(lon, lat)
            else:
                lat2d = lat
                lon2d = lon
            
            # Recortar al área de interés usando el bounding box
            mask = ((lon2d >= bbox["min_lon"]) & (lon2d <= bbox["max_lon"]) &
                    (lat2d  >= bbox["min_lat"]) & (lat2d  <= bbox["max_lat"]))
            if not np.any(mask):
                print(f"No se encontró datos en el bbox para {archivo_in}")
                return False
            
            # Extraer puntos dentro del bbox
            xco2_recortado = xco2[mask].copy()
            lat_recortado = lat2d[mask].copy()
            lon_recortado = lon2d[mask].copy()
            
            # Definir la nueva grilla regular en el área de interés
            lat_min, lat_max = bbox["min_lat"], bbox["max_lat"]
            lon_min, lon_max = bbox["min_lon"], bbox["max_lon"]
            lat_new_vals = np.arange(lat_min, lat_max + resol, resol)
            lon_new_vals = np.arange(lon_min, lon_max + resol, resol)
            # La nueva grilla se genera a partir de los arrays 1D
            lon_new, lat_new = np.meshgrid(lon_new_vals, lat_new_vals)
            
            # Aplicar interpolación por kriging
            interp_data = kriging_interpolation(xco2_recortado, lat_recortado, lon_recortado,
                                                lon_new_vals, lat_new_vals, variogram_model)
            
            # Guardar el archivo de salida en formato NetCDF
            with Dataset(archivo_out, 'w', format='NETCDF4') as ds_out:
                ds_out.createDimension('lat', lat_new.shape[0])
                ds_out.createDimension('lon', lat_new.shape[1])
                
                lat_var = ds_out.createVariable('lat', 'f4', ('lat',))
                lon_var = ds_out.createVariable('lon', 'f4', ('lon',))
                lat_var[:] = lat_new_vals
                lon_var[:] = lon_new_vals
                
                xco2_var = ds_out.createVariable('xco2', 'f4', ('lat', 'lon'))
                xco2_var[:] = interp_data
                
                ds_out.description = (f"Archivo CT recortado e interpolado a 0.05° usando "
                                      f"Ordinary Kriging ({variogram_model} model) para Colombia")
                ds_out.source = archivo_in
            
            print(f"Procesado y guardado: {archivo_out}")
            
            # # Mostrar el plot en pantalla
            # plt.figure(figsize=(8, 6))
            # plt.pcolormesh(lon_new, lat_new, interp_data, shading='auto', cmap='viridis')
            # plt.colorbar(label='xco2')
            # plt.xlabel('Longitud')
            # plt.ylabel('Latitud')
            # plt.title(f"Interpolación Kriging ({variogram_model}) para {os.path.basename(archivo_in)}")
            # plt.tight_layout()
            # plt.show()
            
            return True
    except Exception as e:
        print(f"Error procesando {archivo_in}: {e}")
        return False

if __name__ == '__main__':
    #anio = 2024
    anio = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.datetime.now().year
    
    # Obtener directorio del script actual
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Rutas ABSOLUTAS
    ct_in = os.path.join(SCRIPT_DIR, "..", "v_pre", "Carbon_Tracker")
    carpeta_in = os.path.join(ct_in, str(anio))

    ct_out = os.path.join(SCRIPT_DIR, "Carbon_Tracker") 
    carpeta_out = os.path.join(ct_out, str(anio))
    os.makedirs(carpeta_out, exist_ok=True)

    archivos = os.listdir(carpeta_in)
    total = len(archivos)
    procesados = 0

    variogram_model = 'spherical'  

    for archivo in archivos:
        if archivo.endswith('.nc'):
            ruta_in = os.path.join(carpeta_in, archivo)
            base, ext = os.path.splitext(archivo)
            nombre_out = base + "_co" + ext
            ruta_out = os.path.join(carpeta_out, nombre_out)

            # Verificar si ya existe
            if os.path.exists(ruta_out):
                print(f"⏩ {archivo} ya fue procesado. Omitiendo.")
                continue

            # Procesar si no existe
            if procesar_archivo_ct(ruta_in, ruta_out, BBOX, resolucion, variogram_model):
                procesados += 1

    print(f"✅ Proceso finalizado. Se procesaron {procesados} de {total} archivos.")
