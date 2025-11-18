# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 22:30:10 2025

@author: Zully JBC

Nota: Este código descarga los datos de ERA5, con cada una de las variables
neesarias para el modelo de ERT que estima el fondo del XCO2. 
"""

# 1. Importación de librerias.
import os
import sys
from cdsapi import Client
import xarray as xr
import datetime

# 2. Configuración de carpetas
# Obtener directorio del script actual
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas ABSOLUTAS
CARPETA_OCO2 = os.path.join(SCRIPT_DIR, "..", "..", "datos_OCO2", "OCO2_L2_Lite_FP_Co")
CARPETA_ERA5 = os.path.join(SCRIPT_DIR, "ERA5")  # ← Crea en la misma carpeta del script

# 3. Variables que queremos descargar
VARIABLES_ERA5 = ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind"]

# 5. Descarga de los datos
def descargar_era5(fecha, hora, file_out):
    
    if os.path.exists(file_out):
        print(f"⏩ {os.path.basename(file_out)} ya existe. Omitiendo descarga.")
        return
    
    c = Client()

    año = fecha[:4]
    mes = fecha[4:6]
    dia = fecha[6:8]
    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': VARIABLES_ERA5,
            'year': año,
            'month': mes,
            'day': dia,
            'time': f'{hora[:2]}:00',
            'area': [12.46, -79.01, -4.226, -66.857],  # N, W, S, E (Colombia bounding box)
            'format': 'netcdf',
        },
        file_out
    )
    print(f"📥 ERA5 descargado en: {file_out}")

def main():
    #anio = 2024
    anio = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.datetime.now().year
    carpeta_año_oco2 = os.path.join(CARPETA_OCO2, str(anio))

    carpeta_año_era5 = os.path.join(CARPETA_ERA5, str(anio))
    os.makedirs(carpeta_año_era5, exist_ok=True)

    if not os.path.exists(carpeta_año_oco2):
        print(f"🚫 No existe carpeta de OCO2 para el año {anio}")
        return

    for archivo in sorted(os.listdir(carpeta_año_oco2)):
        if archivo.endswith('.nc4'):
            path_oco2 = os.path.join(carpeta_año_oco2, archivo)
            ds = xr.open_dataset(path_oco2)

            # Extraer fecha y hora del atributo
            mean_datetime = ds.attrs.get('mean_datetime')
            if mean_datetime:
                fecha = mean_datetime[:8]   # YYYYMMDD
                hora = mean_datetime[8:12]  # HHMM

                nombre_salida = f"era5_inst_{fecha}_{hora[:2]}.nc"
                path_era5 = os.path.join(carpeta_año_era5, nombre_salida)

                if os.path.exists(path_era5):
                    print(f"⏩ {nombre_salida} ya existe. Omitiendo descarga.")
                    continue

                descargar_era5(fecha, hora, path_era5)
            else:
                print(f"⚠️ {archivo} no tiene 'mean_datetime'.")

if __name__ == "__main__":
    main()
