# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:47:35 2025

@author: Zully JBC

Nota: Código principal para la generación de los mapas de emisiones de XCO2, asi
como la identificación de los puntos de emisión por medio del modelo inverso de la 
pluma gaussiana.
"""

# ---Importación de librerías---
import sys
import os
import glob
import datetime
import numpy as np
import numpy.ma as ma
import xarray as xr
import rioxarray
import joblib
import pandas as pd
import csv
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import transform
import pyproj
from rasterio.enums import Resampling
from rasterio.features import rasterize
from affine import Affine
from scipy.optimize import lsq_linear
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from gaussian_plume import (
    calcular_baseline_grid,
    detectar_areas_emision,
    construir_matriz_sensibilidad_grid,
    simular_pluma,
    guardar_pluma
)

# ---Configuración de ruta---
INPUT_OCO2 = './datos_OCO2/OCO2_L2_Lite_FP_Co'
INPUT_ERA5 = './variables/v_pos/ERA5'
INPUT_CT = './variables/v_pos/Carbon_Tracker'
INPUT_MODIS_NDVI = './variables/v_pos/MODIS/NDVI'
INPUT_MODIS_LC = './variables/v_pos/MODIS/Land_cover'
INPUT_WPOP = './variables/v_pos/World_pop'
INPUT_SRTM = './variables/v_pos/SRTM/SRTM_Colombia_pos.tif'
SHP_CO = './variables/shp/limite_colombia.shp'


# Carga del modelo entrenado para predecir xco2
raw_model = joblib.load('./modelos_ML/modelo_ert_xco2.joblib')
MODEL_ML = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', raw_model)
])

# ---Funciones de carga de datos rasters y NetCDF---
def load_oco2(file_path):
    """
    Carga un archivo NetCDF de OCO-2 y extrae:
      - xco2: concentración de CO2
      - latitude, longitude: coordenadas de medición
      - ds: Dataset completo para referenciar CRS y metadatos
    """
    ds = xr.open_dataset(file_path)
    co2 = ds['xco2'].values
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    co2_unc = ds['xco2_uncertainty'].values if 'xco2_uncertainty' in ds else None
    return co2, co2_unc, lat, lon, ds


def load_raster(path):
    """
    Abre un ráster con rioxarray y asegura:
      - Sistemas de referencia (EPSG:4326)
      - Dimensiones espaciales definidas (x_dim, y_dim)
    """
    da = rioxarray.open_rasterio(path).squeeze()
    if not da.rio.crs:
        da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim=da.dims[-1], y_dim=da.dims[-2])
    return da


def process_date(o_file):
    """
    Procesa un archivo .nc4 de OCO-2:
      1. Extrae fecha y genera tag YYYYMMDD
      2. Carga variables de OCO-2, ERA5 y Carbon Tracker
      3. Reproyecta y remuestrea rasters auxiliares (NDVI, LandCover, WorldPop, SRTM)
      4. Ensambla un Dataset unificado y aplica Random Forest
      5. Calcula residuales y, si hay datos suficientes, realiza Kriging
      6. Guarda métricas (RMSE, MAE) y resultados ráster
      7. Detecta anomalías y simula plumas gaussianas para zonas de emisión
    """
    # Extraer fecha del nombre de archivo (formato _YYMMDD_)
    basename = os.path.basename(o_file)
    date_part = basename.split('_')[2]
    dt = datetime.datetime.strptime(date_part, '%y%m%d')
    tag = dt.strftime('%Y%m%d')
    
    # Creación de las carpetas de salida
    OUTPUT_DIR_MAP = os.path.join('./resultados/mapas/',str(dt.year))
    os.makedirs(OUTPUT_DIR_MAP, exist_ok=True)
    OUTPUT_DIR_POINT = os.path.join('./resultados/fuentes/',str(dt.year))
    os.makedirs(OUTPUT_DIR_POINT, exist_ok=True)
    
    # Salida de TIF para predicción RF
    out_rf = os.path.join(OUTPUT_DIR_MAP, f'Emisiones_XCO2_{tag}.tif')
    if os.path.exists(out_rf):
        print(f"⏭️ Archivo ya existe para {tag}, se omite.")
        return
    
    # Crear archivo CSV con headers (si no existe)
    CSV_OUT = os.path.join(OUTPUT_DIR_POINT, f'emisiones_co2_{dt.year}.csv')
    if not os.path.exists(CSV_OUT):
        with open(CSV_OUT, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'tag','pluma_id','Q_kgps','sigma_rel',
                'sigma_abs','Q_low95','Q_high95','IQR','IQR_rel', 'Kappa',
                'invers_method', 'c_obs', 'c_wind', 'c_disp', 'confianza', 'Q_reg'
            ])

    # Filtro espacial: carga del shapefile y reproyección---
    gdf_colombia = gpd.read_file(SHP_CO).to_crs("EPSG:4326")
    geom_colombia = gdf_colombia.geometry.union_all()
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3116", always_xy=True).transform
    geom_colombia_proj = transform(project, geom_colombia)
    
    # 1. Carga de datos OCO-2
    co2, co2_unc, lat, lon, ds_oco2 = load_oco2(o_file)
    # Definir ref con CRS y dims para reproyección
    ref = ds_oco2.rio.write_crs("EPSG:4326").rio.set_spatial_dims(
        x_dim="longitude", y_dim="latitude"
    )
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    # 2. Carga y remuestreo de variables meteorológicas ERA5
    era5_files = glob.glob(os.path.join(INPUT_ERA5, str(dt.year), f'*{dt.strftime("%Y%m%d")}*.nc'))
    if not era5_files:
        print(f"⚠️ No se encontró archivo ERA5 para la fecha {dt.strftime('%Y-%m-%d')}")
        print('⏭️ Saltando este archivo y continuando con el siguiente')
        return
    era5_file = era5_files[0]
    ds_era = xr.open_dataset(era5_file)

    # Asignar correctamente coordenadas geográficas reales
    ds_era = ds_era.assign_coords({
        'lat': ds_era['latitude'],
        'lon': ds_era['longitude']
    }).drop_vars(['latitude', 'longitude'])

    # Interpolar a las coordenadas de OCO-2
    points_lat = xr.DataArray(lat_grid.flatten(), dims='points')
    points_lon = xr.DataArray(lon_grid.flatten(), dims='points')

    t2m_interp = ds_era['t2m'].interp(lat=points_lat, lon=points_lon, method='nearest').values.reshape(lat_grid.shape)
    d2m_interp = ds_era['d2m'].interp(lat=points_lat, lon=points_lon, method='nearest').values.reshape(lat_grid.shape)
    u10_interp = ds_era['u10'].interp(lat=points_lat, lon=points_lon, method='nearest').values.reshape(lat_grid.shape)
    v10_interp = ds_era['v10'].interp(lat=points_lat, lon=points_lon, method='nearest').values.reshape(lat_grid.shape)
    sp_interp  = ds_era['sp'].interp(lat=points_lat, lon=points_lon, method='nearest').values.reshape(lat_grid.shape)

    # 3. Carga de CO2 de Carbon Tracker y remuestreo
    ct_files = glob.glob(os.path.join(INPUT_CT, str(dt.year), f'*{dt.strftime("%Y-%m-%d")}*.nc'))
    if not ct_files:
        print(f"⚠️ No se encontró archivo Carbon Tracker para la fecha {dt.strftime('%Y-%m-%d')}")
        print('⏭️ Saltando este archivo y continuando con el siguiente')
        return
    
    ct_file = ct_files[0]
    ds_ct = xr.open_dataset(ct_file).rename({'xco2':'co2_carbon'})
    co2_carbon = ds_ct['co2_carbon'].interp(lat=lat, lon=lon, method="nearest").values
    
    # 4. Carga y reproyección de rásters auxiliares
    # NDVI - Buscar del año actual, si no existe buscar el más reciente
    ndvi_pattern = os.path.join(INPUT_MODIS_NDVI, f'NDVI_{dt.year}_Colombia.tif')
    if not os.path.exists(ndvi_pattern):
        # Buscar el más reciente
        ndvi_files = [f for f in os.listdir(INPUT_MODIS_NDVI) if f.startswith('NDVI_') and f.endswith('.tif')]
        if ndvi_files:
            # Extraer años y encontrar el más reciente
            años_ndvi = []
            for f in ndvi_files:
                try:
                    año = int(f.split('_')[1])  # NDVI_2023_Colombia.tif
                    años_ndvi.append(año)
                except:
                    continue
            if años_ndvi:
                año_reciente_ndvi = max(años_ndvi)
                ndvi_pattern = os.path.join(INPUT_MODIS_NDVI, f'NDVI_{año_reciente_ndvi}_Colombia.tif')
                print(f"⚠️ Usando NDVI del año más reciente disponible: {año_reciente_ndvi}")
    
    # Land Cover - Misma lógica
    lc_pattern = os.path.join(INPUT_MODIS_LC, f'LandCover_Colombia_{dt.year}.tif')
    if not os.path.exists(lc_pattern):
        lc_files = [f for f in os.listdir(INPUT_MODIS_LC) if f.startswith('LandCover_Colombia_') and f.endswith('.tif')]
        if lc_files:
            años_lc = []
            for f in lc_files:
                try:
                    año = int(f.split('_')[2].split('.')[0])  # LandCover_Colombia_2023.tif
                    años_lc.append(año)
                except:
                    continue
            if años_lc:
                año_reciente_lc = max(años_lc)
                lc_pattern = os.path.join(INPUT_MODIS_LC, f'LandCover_Colombia_{año_reciente_lc}.tif')
                print(f"⚠️ Usando Land Cover del año más reciente disponible: {año_reciente_lc}")
    
    # WorldPop - Misma lógica
    wpop_pattern = os.path.join(INPUT_WPOP, f'*{dt.year}*.tif')
    wpop_files = glob.glob(wpop_pattern)
    if not wpop_files:
        # Buscar cualquier archivo WorldPop
        wpop_files = glob.glob(os.path.join(INPUT_WPOP, '*.tif'))
        if wpop_files:
            # Encontrar el más reciente por el año en el nombre
            años_wpop = []
            for f in wpop_files:
                try:
                    # Buscar cualquier número de 4 dígitos en el nombre
                    import re
                    años = re.findall(r'\d{4}', os.path.basename(f))
                    if años:
                        años_wpop.append(int(años[0]))
                except:
                    continue
            if años_wpop:
                año_reciente_wpop = max(años_wpop)
                wpop_pattern = os.path.join(INPUT_WPOP, f'*{año_reciente_wpop}*.tif')
                wpop_files = glob.glob(wpop_pattern)
                print(f"⚠️ Usando WorldPop del año más reciente disponible: {año_reciente_wpop}")
    
    # Verificar que los archivos existen después de la búsqueda
    if not os.path.exists(ndvi_pattern):
        print(f"⚠️ No se encontró archivo NDVI para {dt.year} ni años anteriores")
        return
        
    if not os.path.exists(lc_pattern):
        print(f"⚠️ No se encontró archivo Land Cover para {dt.year} ni años anteriores")
        return
        
    if not wpop_files:
        print(f"⚠️ No se encontró archivo WorldPop para {dt.year} ni años anteriores")
        return
    
    # Cargar rásters
    ndvi = load_raster(ndvi_pattern)
    lc = load_raster(lc_pattern)
    wpop = load_raster(wpop_files[0])
    srtm = load_raster(INPUT_SRTM)

    layers = {}
    for name, da in [('ndvi', ndvi), ('land_cover', lc), ('world_pop', wpop), ('srtm', srtm)]:
        arr = da.rio.reproject_match(ref, resampling=Resampling.nearest).values  # Usar resampling nearest
        layers[name] = arr

    # 5. Ensamble de Dataset unificado con variables predictoras
    ds_all = xr.Dataset({
        'lat': ('lat', np.unique(lat_grid)),  # Usar valores únicos 1D
        'lon': ('lon', np.unique(lon_grid)),
        't2m': (('lat','lon'), t2m_interp),
        'd2m': (('lat','lon'), d2m_interp),
        'u10': (('lat','lon'), u10_interp),
        'v10': (('lat','lon'), v10_interp),
        'sp': (('lat','lon'), sp_interp),
        'co2_carbon': (('lat','lon'), co2_carbon),
        'srtm': (('lat','lon'), layers['srtm']),
        'ndvi': (('lat','lon'), layers['ndvi']),
        'land_cover': (('lat','lon'), layers['land_cover']),
        'world_pop': (('lat','lon'), layers['world_pop'])
    })
    
    feature_names = [
    'lat', 'lon', 't2m', 'd2m', 'sp', 
    'u10', 'v10', 'co2_carbon', 'srtm', 
    'ndvi','land_cover', 'world_pop']
    
    # Aplanar grilla y apilar dimensiones en 'points'
    ds_stack = ds_all.stack(points=('lat','lon'))
    X = pd.DataFrame({name: ds_stack[name].values for name in feature_names})

    # Validar si hay columnas completamente nulas
    empty_cols = X.columns[X.isna().all()].tolist()
    if empty_cols:
         print(f"⚠️ Las siguientes columnas están completamente vacías: {empty_cols}")
         X = X.drop(columns=empty_cols)
         print('⏭️ Se pasará a los siguientes archivos')
         return
        
    # 6. Predicción con el modelo de ML
    MODEL_ML.named_steps['imputer'].fit(X)
    X_imputed = MODEL_ML.named_steps['imputer'].transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    y_pred = MODEL_ML.named_steps['model'].predict(X_imputed).reshape(lat_grid.shape)
   
    # Guardar resultado final como ráster GeoTIFF
    result_array = xr.DataArray(
        y_pred,
        dims=('latitude', 'longitude'),
        coords={'latitude': lat, 'longitude': lon}
    )
    result_array = result_array.rio.write_crs("EPSG:4326")
    result_array = result_array.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    out_rf = os.path.join(OUTPUT_DIR_MAP, f'Emisiones_XCO2_{tag}.tif')
    result_array.rio.to_raster(out_rf, dtype='float32')
    print(f"✅ Se guardó el mapa correctamente como: Emisiones_XCO2_{tag}.tif")
    
    # 7. Simulación de pluma gaussiana en áreas de anomalía
    co2_clean = y_pred
    baseline, anom = calcular_baseline_grid(co2_clean)
    anom_values = co2_clean - baseline 

    if not np.any(anom):
        print(f"⚠️ No se detectaron anomalías en {tag}, se omite el archivo.")
        return
    
    # Interpolar ERA5 a la cuadrícula de OCO-2
    ds_era5 = xr.open_dataset(era5_file)
    lat_vals = ds_era5['latitude']
    lon_vals = ds_era5['longitude']
    
    # Limpiar variables duplicadas y reasignar coords lat/lon
    for var in ['latitude', 'longitude', 'lat', 'lon']:
        if var in ds_era5:
            ds_era5 = ds_era5.drop_vars(var)
        if var in ds_era5.coords:
            ds_era5 = ds_era5.reset_coords(var, drop=True)
    
    ds_era5 = ds_era5.assign_coords({
        'lat': lat_vals,
        'lon': lon_vals
    })
    
   
    ds_era_interp = ds_era5.interp(
        lat = xr.DataArray(lat, dims="lat"),
        lon = xr.DataArray(lon, dims="lon"),
        method = "nearest"
        )
   
    # Detección de polígonos de emisión y rasterización
    areas_polygons = detectar_areas_emision(anom, lat_grid, lon_grid)

    dx = lon[1] - lon[0]  
    dy = lat[1] - lat[0]
    transformar = Affine(dx, 0, lon.min() - dx/2,
                      0, -dy, lat.max() + dy/2)
    
    masks = []
    for poly in areas_polygons:
        if poly is None or poly.is_empty or not poly.is_valid:
            continue
        
        # Convertir a coordenadas de píxel
        mask = rasterize(
            [poly],
            out_shape=anom.shape,
            transform=transformar,
            fill=0,
            all_touched=True,  # Incluir píxeles que tocan el polígono
            dtype=np.uint8
        )
        if np.sum(mask) > 0:
            masks.append(mask.astype(bool))
        else:
            print(f"⚠️ Polígono no rasterizado: Área={poly.area:.6f}, Bounds={poly.bounds}")
    

    print(f"Número de áreas: {len(masks)}")
    
    if not masks or all(np.sum(mask) == 0 for mask in masks):
        print(f"⚠️ No se detectaron áreas de emisión en {o_file}, salto al siguiente.")
        return

    # 8. Inversión de emisiones Q 
    A = construir_matriz_sensibilidad_grid(masks, lat_grid, lon_grid, ds_era_interp, anom_values)
    b = (co2_clean - baseline).flatten()
    
    if ma.isMaskedArray(b):
        b = b.filled(0)  # Rellenar NaN con 0
    if ma.isMaskedArray(A):
        A = A.filled(0)
    
    # Verificar datos no nulos
    if np.all(A == 0) or np.all(b == 0):
        print("⚠️ Todos los valores en A o b son cero. Plumas descartadas.")
        Q_estimado = np.full(len(masks), np.nan)
    else:
        try:
            result = lsq_linear(A, b, bounds=(0, np.inf), max_iter=1000)
            Q_estimado = result.x
        except ValueError as e:
            print(f"🛑 Error en inversión: {e}. Plumas descartadas.")
            Q_estimado = np.full(len(masks), np.nan)

    # 9. Simular plumas con Q real 
    emission_field = np.zeros_like(lat_grid, dtype=float)
    
    for i, mask in enumerate(masks):
        try:
            if np.isnan(Q_estimado[i]):
               print(f"⚠️ Pluma #{i} descartada: Q no se pudo estimar.")
               continue
           
            if Q_estimado[i] <= 0.02:
                print(f"⚠️ Pluma #{i} descartada: Q demasiado bajo ({Q_estimado[i]:.2e})")
                continue
        
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
        
            # - Punto de mayor anomalía dentro del polígono
            ani_vals = anom[mask]
            max_idx = np.nanargmax(ani_vals)
            y0, x0 = ys[max_idx], xs[max_idx]
            lat0, lon0 = lat_grid[y0, x0], lon_grid[y0, x0] 
        
            # - Footprint: círculo de 5 km alrededor del hotspot
            dx = (lon_grid - lon0) * 111320 * np.cos(np.deg2rad(lat0))
            dy = (lat_grid - lat0) * 111320
            circle_mask = dx**2 + dy**2 <= (5e3)**2
            source_mask = mask & circle_mask
        
            if not np.any(source_mask):
                print(f"⚠️ Footprint 5 km no tiene datos en pluma #{i}, salto.")
                continue
        
            # - Velocidad y dirección del viento
            u = ds_era_interp['u10'].values[y0, x0]
            v = ds_era_interp['v10'].values[y0, x0]
            velocidad = np.sqrt(u**2 + v**2)
            direccion = (np.arctan2(-u, -v) * 180 / np.pi) % 360
        
            # - Filtro de tiempo de residencia
            # Determinar clase de estabilidad nocturna según Pasquill
            if velocidad >= 5:
                stability = 'D'
            elif velocidad < 2:
                stability = 'F'
            elif velocidad < 3:
                stability = 'E'
            else:
                stability = 'D'  # intermedio
    
            # Establecer t_max según clase
            t_max_dict = {'F': 4 * 3600, 'E': 3 * 3600, 'D': 2 * 3600}
            t_max = t_max_dict.get(stability, 2 * 3600)
    
            # Garantizar mínimo de 2h si el viento es muy débil
            if velocidad < 0.5:
                t_max = max(t_max, 2 * 3600)
    
            #print(f"🌀 Velocidad={velocidad:.2f} m/s, Estabilidad={stability}, t_max set a {t_max/3600:.1f}h")
                
            # Calculamos tiempo de residencia
            theta = np.radians(180 + direccion)
            dx = (lon_grid - lon0) * 111320 * np.cos(np.deg2rad(lat0))
            dy = (lat_grid - lat0) * 111320
            x_rot = dx * np.cos(theta) - dy * np.sin(theta)
            
            mask_distancia = mask & (x_rot > 0)
            t_secs = x_rot[mask_distancia] / velocidad if velocidad > 0 else np.array([])
            
            if t_secs.size == 0 or np.all(t_secs > t_max):
                print(f"⚠️ Pluma #{i} descartada: tiempo de residencia > {t_max/3600:.1f}h")
                continue
        
            # Simular y sumar pluma
            pluma = simular_pluma(
                lon_grid, lat_grid,
                lon0, lat0, Q_estimado[i],
                velocidad, direccion
            )
            
            # Filtro espacial por shp
            lon_pluma = lon_grid[ys.min():ys.max()+1, xs.min():xs.max()+1]
            lat_pluma = lat_grid[ys.min():ys.max()+1, xs.min():xs.max()+1]
            geom = box(
                lon_pluma.min(), lat_pluma.min(),
                lon_pluma.max(), lat_pluma.max()
            )
            geom_proj = transform(project, geom)
            inter = geom_proj.intersection(geom_colombia_proj)
            frac_dentro = inter.area / geom_proj.area if geom_proj.area > 0 else 0

            if frac_dentro < 0.7:
                print(f"⚠️ Pluma #{i} descartada: solo {frac_dentro:.1%} dentro de Colombia")
                continue


            emission_field += pluma
            
            # Propagación de errores (Budget híbrido + diagnóstico cond(A))
            # A) Error de concentración
            
            # Concentración promedio de fondo (ppm) en la pluma
            conc_fondo = np.nanmean(co2_clean[mask])
        
            # Error absoluto (ppm) = combinación en cuadratura de satélite + ML
            sigma_conc_abs = np.sqrt(0.75**2 + 0.785**2)  # ≈ 1.07 ppm
            
            # Error relativo respecto al fondo
            OBS_ERR_REL = sigma_conc_abs / conc_fondo  
            
            # B) Error de viento
            sigma_bias = 0.5  # m/s
            sigma_error = 1.0 # m/s
            sigma_total = (sigma_bias**2 + sigma_error**2)**0.5  # ≈1.12 m/s
            
            wind_factor = sigma_total / max(velocidad, 0.5)
            wind_factor = min(wind_factor, 0.38)
 

            # C) Error de dispersión (según estabilidad de Pasquill)
            disp_err_by_class = {
                'D': 0.10,
                'E': 0.20,
                'F': 0.30
            }
            ERR_DISP = disp_err_by_class.get(stability, 0.20)  # Default 20% si no se reconoce
          
            if stability == 'F' and velocidad < 2.0:
                ERR_DISP = max(ERR_DISP, 0.35)

            # Error acumulado total
            sigma_rel_total = np.sqrt(
                OBS_ERR_REL**2 +
                wind_factor**2 +
                ERR_DISP**2
            )
            sigma_abs_total = sigma_rel_total * Q_estimado[i]
            
            Q_low_95 = Q_estimado[i] - 1.96 * sigma_abs_total
            Q_high_95 = Q_estimado[i] + 1.96 * sigma_abs_total
            IQR = 1.349 * sigma_abs_total
            IQR_rel = (IQR / Q_estimado[i]) if Q_estimado[i] > 0 else np.nan
            
            # --- Contribuciones relativas (decimales) ---
            c_obs  = (OBS_ERR_REL**2) / (sigma_rel_total**2)
            c_wind = (wind_factor**2)  / (sigma_rel_total**2)
            c_disp = (ERR_DISP**2)     / (sigma_rel_total**2)
            
            
            # --- Diagnóstico cond(A) ---
            try:
                kappa = np.linalg.cond(A)
            except Exception:
                kappa = np.nan
           
            # --- Nivel de confianza ---
            if kappa <= 10:
                confianza = "alta"
            elif kappa <= 30:
                confianza = "moderada"
            else:
                confianza = "baja"
            
            # --- Reintento con regularización si cond(A)>30 ---
            Q_reg = np.nan
            inversion_method = "lsq_linear"
            if kappa > 30:
                try:
                    # Truncated SVD
                    U, s, Vt = np.linalg.svd(A, full_matrices=False)
                    tol = s.max() * 1e-6
                    r = max(1, np.sum(s > tol))
                    s_inv = np.array([1/si if si > tol else 0 for si in s])
                    Q_reg = (Vt.T * s_inv).dot(U.T.dot(b))
                    inversion_method = f"tsvd(r={r})"
                except Exception:
                    try:
                        # Tikhonov
                        smax = s[0] if len(s) > 0 else 1.0
                        lambda_reg = (smax * 1e-3) if smax > 0 else 1e-6
                        ATA = A.T.dot(A)
                        ATb = A.T.dot(b)
                        reg = (lambda_reg**2) * np.eye(ATA.shape[0])
                        Q_reg = np.linalg.solve(ATA + reg, ATb)
                        inversion_method = f"tikhonov(lambda={lambda_reg:.1e})"
                    except Exception as e:
                        print(f"⚠️ Regularización fallida: {e}")
                        inversion_method = "failed_all"
            
            # --- Debug ---
            print(f"Pluma #{i} | Q={Q_estimado[i]:.3f} kg/s | vel={velocidad:.2f} m/s")
            print(f"  - Obs err={OBS_ERR_REL:.2f}, Wind factor={wind_factor:.2f}, Disp={ERR_DISP:.2f}")
            print(f"  => sigma_rel_total={sigma_rel_total:.2f} ({sigma_rel_total*100:.1f}%)")
            print(f"  - cond(A)={kappa:.1f}, confianza={confianza}, metodo={inversion_method}")
            
            # --- Guardar en CSV ---
            with open(CSV_OUT, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # Si Q_reg es una lista/array, convertir a string compacto
                if isinstance(Q_reg, (list, np.ndarray)):
                    Q_reg = "[" + " ".join(f"{x:.8f}" for x in Q_reg) + "]"
                
                writer.writerow([
                    tag, i, Q_estimado[i],
                    sigma_rel_total, sigma_abs_total,
                    Q_low_95, Q_high_95, IQR, IQR_rel,
                    kappa, inversion_method,
                    c_obs, c_wind, c_disp,
                    confianza, Q_reg
                ])
            
            # Guardar pluma por separado
            guardar_pluma(pluma, lat, lon, tag, i, OUTPUT_DIR_POINT)

        except Exception as e:
            print(f"🛑 Error en pluma #{i}: {e}")
            continue
    

def main(year=None):
    if not year:
        year = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.datetime.now().year
    files = sorted(
        glob.glob(os.path.join(INPUT_OCO2, str(year), '*.nc4')))
    if not files:
        print(
            f'⚠️ No se encontraron archivos para el año {year} ' \
            f'en {os.path.join(INPUT_OCO2, str(year))}'
        )
        return
    for f in files:
        process_date(f)
    print(f"Se procesaron todos los archivos del {year}")

if __name__ == '__main__':
    main()
