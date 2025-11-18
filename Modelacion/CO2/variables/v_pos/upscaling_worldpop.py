# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:04:05 2025

@author: Zully JBC

Nota: Código para hacer upscaling de los datos de World Pop, cuya resolución espacial
inicial es de 1km y sube a 0.05° por medio de interpolación bilineal. 
"""

# 1. Importación de librerias
import os
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio import warp
import datetime

# 2. Parámetros de la grilla para Colombia
target_lat_min = -4.226
target_lat_max = 12.46
target_lon_min = -79.01
target_lon_max = -66.857
res = 0.05
target_crs = "EPSG:4326"

target_transform = from_origin(target_lon_min, target_lat_max, res, res)
target_width = int(round((target_lon_max - target_lon_min) / res))
target_height = int(round((target_lat_max - target_lat_min) / res))

# 3. Función de remuestreo
def resample_to_target(input_raster_path, output_raster_path, resampling_method=Resampling.nearest):
    with rasterio.open(input_raster_path) as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': target_transform,
            'width': target_width,
            'height': target_height
        })
        os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
        with rasterio.open(output_raster_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=resampling_method
                )
    print(f"✅ Archivo remuestreado: {output_raster_path}")

# 🧠 Main
if __name__ == "__main__":
    #anio = 2024
    anio = datetime.datetime.now().year
    carpeta_entrada = "../v_pre/World_pop"
    carpeta_salida = "./World_pop"
    os.makedirs(carpeta_salida, exist_ok=True)

    # 🧐 Posibles nombres
    posibles_archivos = [
        f"col_ppp_{anio}_1km_Aggregated.tif",
        f"col_landscan_global_{anio}.tif"
    ]

    encontrado = False
    for archivo in posibles_archivos:
        ruta_entrada = os.path.join(carpeta_entrada, archivo)
        if os.path.exists(ruta_entrada):
            ruta_salida = os.path.join(carpeta_salida, f"world_pop_{anio}_005deg.tif")
            resample_to_target(ruta_entrada, ruta_salida, Resampling.bilinear)
            encontrado = True
            break  

    if not encontrado:
        print(f"🚫 No se encontró ningún archivo de población para el año {anio}. No hay nada que remuestrear.")
