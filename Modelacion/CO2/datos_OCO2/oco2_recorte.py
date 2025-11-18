# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 18:58:58 2025

@author: Zully JBC

Nota: Este código se diseño para recortar y descartar los archivos de OCO-2 de la NASA
que no cubran a Colombia con datos del XCO; además cambia su resolución a 0.05°, utilizando
bining, de mantener la estructura grillada de los datos y generar sesgos de interpolación
al contar con poca densisdad de puntos. El dato queda listo para ser interpolado con 
kriging de residuos, después de la predicción del xco2 de Random Forest con las variables 
previamente definidas. 
"""

# ---Importación de librerías---
import os
import sys
import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_2d
from datetime import datetime  # para año dinámico

# ---Configuración de carpetas---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Obtener directorio del script actual
CARPETA_ENTRADA = os.path.join(SCRIPT_DIR,"OCO2_L2_Lite_FP")
CARPETA_SALIDA = os.path.join(SCRIPT_DIR,"OCO2_L2_Lite_FP_Co")
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# ---Bounding box para Colombia---
lat_min, lat_max = -4.226, 12.46
lon_min, lon_max = -79.01, -66.857

# ---Resolución deseada en grados para el binning---
res = 0.05

def binning_a_grilla(ds):
    df = ds[['latitude', 'longitude', 'xco2', 'xco2_uncertainty']].to_dataframe().dropna().reset_index(drop=True)
    df = df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
            (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)]
    if df.empty:
        return None

    lat_bins = np.arange(lat_min, lat_max + res, res)
    lon_bins = np.arange(lon_min, lon_max + res, res)

    # binning de xco2 y de incertidumbre
    stat_xco2, lat_edge, lon_edge, _ = binned_statistic_2d(
        df['latitude'], df['longitude'], df['xco2'],
        statistic='mean', bins=[lat_bins, lon_bins]
    )
    stat_unc, _, _, _ = binned_statistic_2d(
        df['latitude'], df['longitude'], df['xco2_uncertainty'],
        statistic='mean', bins=[lat_bins, lon_bins]
    )

    lat_centers = lat_edge[:-1] + res/2
    lon_centers = lon_edge[:-1] + res/2

    ds_out = xr.Dataset(
        {
            'xco2': (('latitude', 'longitude'), stat_xco2),
            'xco2_uncertainty': (('latitude', 'longitude'), stat_unc)
        },
        coords={'latitude': lat_centers, 'longitude': lon_centers}
    )
    return ds_out


def procesar_archivo(netcdf_file, file_out):
    try:
        if os.path.exists(file_out):
            print(f"⏩ {os.path.basename(netcdf_file)}: Ya procesado. Omitiendo.")
            return 0

        ds = xr.open_dataset(netcdf_file, decode_times=False)
        bbox_mask = ((ds.latitude >= lat_min) & (ds.latitude <= lat_max) &
                     (ds.longitude >= lon_min) & (ds.longitude <= lon_max))
        ds_filtrado = ds.where(bbox_mask, drop=True)

        if ds_filtrado['xco2'].count() == 0:
            print(f"🚫 {os.path.basename(netcdf_file)}: Sin datos de xco2 en Colombia.")
            return 0

        if 'xco2_quality_flag' in ds_filtrado:
            ds_filtrado = ds_filtrado.where(ds_filtrado['xco2_quality_flag'] == 0, drop=True)
        else:
            print(f"⚠️ {os.path.basename(netcdf_file)}: Sin 'xco2_quality_flag'. Continúo sin filtro.")

        if ds_filtrado['xco2'].count() == 0:
            print(f"🚫 {os.path.basename(netcdf_file)}: Sin datos tras filtro de calidad.")
            return 0

        ds_binned = binning_a_grilla(ds_filtrado)
        if ds_binned is None:
            print(f"⛔ {os.path.basename(netcdf_file)}: Binning sin datos en el área.")
            return 0

        ids = ds_filtrado['sounding_id'].values
        fecha_media = str(int(np.mean([int(x) for x in ids])))[:12]  # YYYYMMDDHHMM
        ds_binned.attrs['mean_datetime'] = fecha_media

        os.makedirs(os.path.dirname(file_out), exist_ok=True)
        ds_binned.to_netcdf(file_out)
        print(f"✅ {os.path.basename(file_out)}: Procesado OK.")
        return 1

    except Exception as e:
        if "dimension 'epoch_dimension'" in str(e):
            print(f"❌ {os.path.basename(netcdf_file)}: Vacío (error de dimensiones).")
        else:
            print(f"❌ {os.path.basename(netcdf_file)}: Error → {e}")
        return 0

def procesar_anio(anio):
    """
    Procesa todos los .nc4 de un año y devuelve cuántos archivos válidos procesó.
    """
    carpeta_anio_in = os.path.join(CARPETA_ENTRADA, str(anio))
    carpeta_anio_out = os.path.join(CARPETA_SALIDA, str(anio))

    if not os.path.isdir(carpeta_anio_in):
        print(f"🚫 No existe carpeta para el año {anio}.")
        return 0

    archivos = sorted([f for f in os.listdir(carpeta_anio_in) if f.endswith('.nc4')])
    if not archivos:
        print(f"🚫 No hay archivos .nc4 en {anio}.")
        return 0

    total_procesados = 0
    for fname in archivos:
        in_path = os.path.join(carpeta_anio_in, fname)
        out_fname = os.path.splitext(fname)[0] + "_co.nc4"
        out_path = os.path.join(carpeta_anio_out, out_fname)
        procesados = procesar_archivo(in_path, out_path)
        total_procesados += procesados

    print(f"🎉 Procesados {total_procesados} archivos para el {anio}.")
    return total_procesados

def main(anio=None, año_min=2020):
    # Si no das año, toma el actual
    if anio is None:
        anio = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.now().year

    # Intenta hasta encontrar datos o llegar al año_min
    while anio >= año_min:
        print(f"\n>>> Probando año {anio}...")
        procesados = procesar_anio(anio)
        if procesados > 0:
            print(f"✅ ¡Listo! Se ya se procesaron todos los datos del {anio}.")
            break
        else:
            print(f"🔄 Nada en {anio}, retrocedo a {anio-1}.")
            anio -= 1
    else:
        print(f"❌ No se encontraron datos desde {datetime.now().year} hasta {año_min}.")

if __name__ == "__main__":
    main()
