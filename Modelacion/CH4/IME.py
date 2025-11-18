# -*- coding: utf-8 -*-
"""
Created on Sun May 18 18:57:32 2025

@author: Zully JBC

Nota: Código principal para la identificación de puntos de emisión de CH4 por 
medio del método IME, utilizando los datos del Sentinel 5P en agrupaciones 
temporales de 5 días.
"""

import sys
import os
import datetime
import numpy as np
import rasterio
import csv
import xarray as xr
import cdsapi
import warnings
import traceback
import pyproj
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import transform
from scipy.stats import ttest_ind
from scipy.ndimage import median_filter, gaussian_filter
from scipy.ndimage import binary_opening, binary_closing
from scipy.interpolate import RegularGridInterpolator
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from rasterio.warp import transform as rio_transform
from skimage.measure import regionprops, label
from sklearn.linear_model import LinearRegression

# Configuración de carpetas
BASE_SENTINEL = "./datos_CH4/SENTINEL_5P_L2"
BASE_ERA5     = "./variables/ERA5"
BASE_RESULT   = "./resultados"
DEM_PATH = './variables/SRTM/SRTM_Colombia.tif' 

# Carga de shp de Colombia
SHP_CO = './variables/shp/limite_colombia.shp'
gdf_colombia = gpd.read_file(SHP_CO).to_crs("EPSG:4326")
geom_colombia = gdf_colombia.geometry.union_all()
project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3116", always_xy=True).transform
geom_colombia_proj = transform(project, geom_colombia)


# ---Parámetros ajustables---
ANOM_THRESHOLD  = 25     # ppb mínimo anomalía
Q_THRESHOLD     = 0.01   # kg/s mínimo flujo Q para considerar
AREA_FRAC_MIN   = 0.05   # fracción mínima de píxeles
MIN_ELON_RATIO  = 2      # elongación mínima
TT_ALPHA        = 0.05   # nivel de significancia t-test
WINDOW_SIZE     = 5      # ventana para t-test
MEDIAN_SIZE     = 5      # filtro de mediana
GAUSS_SIGMA     = 1.5    # sigma gaussiano

def registrar_procesamiento(fn, out_dir, num_plumas):
    """
    Marca que el archivo ya fue procesado
    
    Args:
        fn (str): nombre del archivo procesado
        out_dir (str): directorio de resultados
        num_plumas (int): número de plumas detectadas
    """
   
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "procesados.txt")
    with open(path, "a") as f:
        f.write(f"{fn},{num_plumas}\n")

def calcular_ueff(U10, alpha1=1.0, alpha2=0.6):
    """
    Calcula velocidad de viento efectiva Ueff a partir de U10

    Args:
        U10 (float): velocidad del viento a 10 metros (m/s)
        alpha1 (float): parámetro de calibración (default 1.0)
        alpha2 (float): parámetro de calibración (default 0.6)
    
    Returns:
        float: velocidad de viento efectiva (m/s)
    """
    return alpha1 * np.log(U10 + 1e-3) + alpha2

def calcular_Q(IME, L, U10_series, alpha1=1.0, alpha2=0.6, max_iter=10):
    """
    Calcula la tasa de emisión Q usando el método IME iterativo

    Args:
        IME (float): exceso integrado de CH₄ (kg)
        L (float): longitud de la pluma (m)
        U10_series (np.array): serie temporal de viento a 10 m (m/s)
        alpha1 (float): parámetro de calibración de Ueff
        alpha2 (float): parámetro de calibración de Ueff
        max_iter (int): número máximo de iteraciones para convergencia
    
    Returns:
        tuple:
            Q (float): tasa de emisión estimada (kg/s)
            Ueff (float): velocidad de viento efectiva (m/s)
            tau (float): tiempo de residencia de la pluma (s)
    """
    
    #print(f"    -> calcular_Q entrada: IME={IME:.1f} kg, L={L:.1f} m, pasos viento={len(U10_series)}")
    tau = L / calcular_ueff(np.mean(U10_series), alpha1, alpha2)
    for _ in range(max_iter):
        N = int(min(len(U10_series), np.round(tau)))
        if N < 1:
            break
        U10_avg = np.mean(U10_series[:N])
        Ueff = calcular_ueff(U10_avg, alpha1, alpha2)
        tau_new = L / Ueff
        if abs(tau_new - tau) / tau < 0.01:
            tau = tau_new
            break
        tau = tau_new
    Q = IME / tau #kg/s
    return Q, Ueff, tau

def crear_mascara_pluma(anom, bg_vals):
    """
    Aplica un t-test local sobre la matriz de anomalías para identificar zonas con 
    diferencia significativa.
    
    Args:
        anom (np.array): matriz 2D de anomalías de CH₄ (ppb)
        bg_vals (np.array): valores de fondo (ppb) tomados fuera de la pluma
    
    Returns:
        np.array: máscara binaria booleana con zonas significativas (True donde hay pluma)
    """
    ny, nx = anom.shape
    pad = WINDOW_SIZE // 2
    mask = np.zeros_like(anom, dtype=bool)
    for y in range(pad, ny - pad):
        for x in range(pad, nx - pad):
            ventana = anom[y-pad:y+pad+1, x-pad:x+pad+1].ravel()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p = ttest_ind(ventana, bg_vals, equal_var=False, nan_policy='omit')
            mask[y, x] = (p < TT_ALPHA)
    # filtros morfológicos
    mask = binary_opening(mask, structure=np.ones((3,3)))
    mask = binary_closing(mask, structure=np.ones((3,3)))
    mask = median_filter(mask.astype(int), size=MEDIAN_SIZE)
    mask = gaussian_filter(mask.astype(float), GAUSS_SIGMA) > 0.5
    return mask

def es_pluma_alargada(mask):
    """
    Evalúa si la región más grande en la máscara tiene elongación suficiente para 
    considerarse pluma.
    
    Args:
        mask (np.array): máscara binaria con la forma de la pluma
    
    Returns:
        bool: True si la región más grande tiene elongación y solidez suficientes, 
        False si no
    """
    etiquetas = label(mask)
    props = regionprops(etiquetas)
    if not props:
        return False
    mayor = max(props, key=lambda p: p.area)
    if mayor.minor_axis_length < 5:
        return False
    elongacion = mayor.major_axis_length / mayor.minor_axis_length
    return (elongacion >= MIN_ELON_RATIO) and (mayor.solidity > 0.7)

def pre_filtro_tif(in_tif, MIN_AREA_KM2=200.0, ANOM_UMBRAL=20.0):
    """
    Pre-filtro rápido para descartar tifs casi vacíos antes de procesar.
    
    Args:
        in_tif (str): ruta al GeoTIFF
        MIN_AREA_KM2 (float): área mínima de región para considerarla válida (km²)
        ANOM_UMBRAL (float): anomalía mínima para considerar píxel activo (ppb)
    
    Returns:
        bool: True si el tif tiene al menos una región válida, False si se descarta
    """
    with rasterio.open(in_tif) as src:
        ch4 = src.read(1).astype(float)
        tf = src.transform

    # Cálculo de área promedio por píxel (aprox. en km²)
    pix_w_deg = abs(tf.a)
    pix_h_deg = abs(tf.e)
    lat_mean = np.nanmean(np.linspace(-90, 90, ch4.shape[0]))  # aproximación rápida
    deg2m_lat = 110574.0
    deg2m_lon = 111320.0 * np.cos(np.deg2rad(lat_mean))
    pixel_area_km2 = (pix_w_deg * deg2m_lon) * (pix_h_deg * deg2m_lat) / 1e6
    MIN_AREA_PX = max(3, int(np.ceil(MIN_AREA_KM2 / pixel_area_km2)))

    # Fondo simple y anomalía
    fondo_simple = np.nanpercentile(ch4, 5)
    anom_simple = ch4 - fondo_simple

    # Píxeles que superan umbral mínimo
    mask = (anom_simple > ANOM_UMBRAL) & ~np.isnan(anom_simple)

    # Etiquetar regiones conectadas y filtrar por tamaño
    etiquetas = label(mask, connectivity=2)
    num_regiones = etiquetas.max()
    for reg_id in range(1, num_regiones + 1):
        if (etiquetas == reg_id).sum() >= MIN_AREA_PX:
            return True  # al menos una región válida

    # Si llegamos acá, no hay regiones válidas
    return False

def interpolar_viento_a_grilla(u_era, v_era, shape_sent):
    """
    Interpola los campos de viento ERA5 (u y v) a la grilla del Sentinel-5P.
    
    Args:
        u_era (np.array): componente zonal del viento (lat, lon)
        v_era (np.array): componente meridional del viento (lat, lon)
        shape_sent (tuple): dimensiones (rows, cols) de la grilla Sentinel-5P
    
    Returns:
        tuple:
            U_sent (np.array): componente u interpolada a la grilla Sentinel
            V_sent (np.array): componente v interpolada a la grilla Sentinel
    """
    if u_era.ndim == 3:
        u_era = u_era[0]
    if v_era.ndim == 3:
        v_era = v_era[0]

    lat_era = np.linspace(13, -5, u_era.shape[0])
    lon_era = np.linspace(-80, -66, u_era.shape[1])

    interp_u = RegularGridInterpolator((lat_era, lon_era), u_era, bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator((lat_era, lon_era), v_era, bounds_error=False, fill_value=None)

    rows, cols = np.indices(shape_sent)
    dummy_tf = rasterio.transform.from_origin(-80, 13, (14/shape_sent[1]), (18/shape_sent[0]))
    xs, ys = dummy_tf * (cols, rows)
    lons, lats = xs, ys
    points = np.stack([lats.ravel(), lons.ravel()], axis=-1)

    U_sent = interp_u(points).reshape(shape_sent)
    V_sent = interp_v(points).reshape(shape_sent)
    return U_sent, V_sent

def obtener_era5_u_v(fecha):
    """
    Descarga y carga los datos diarios de viento u10 y v10 de ERA5 para una fecha 
    específica.

    Args:
        fecha (datetime.date): fecha de interés
    
    Returns:
        xr.Dataset: Dataset combinado con variables u10 y v10 como xarray DataArrays
    """

    year_dir = os.path.join(BASE_ERA5, str(fecha.year))
    os.makedirs(year_dir, exist_ok=True)
    u_path = os.path.join(year_dir, f"era5_u10_{fecha}.nc")
    v_path = os.path.join(year_dir, f"era5_v10_{fecha}.nc")
    ps_path = os.path.join(year_dir, f"era5_sp_{fecha}.nc")

    cds = cdsapi.Client()

    req_base = {
        "product_type": "reanalysis",
        "daily_statistic": "daily_mean",
        "frequency": "3_hourly",
        "time_zone": "utc-05:00",
        "area": [13.0, -80.0, -5.0, -66.0],
        "year": str(fecha.year),
        "month": f"{fecha.month:02d}",
        "day": f"{fecha.day:02d}",
        "format": "netcdf"
    }

    if not os.path.exists(u_path):
        cds.retrieve("derived-era5-single-levels-daily-statistics",
                     {**req_base, "variable": ["10m_u_component_of_wind"]}, u_path)
    
    if not os.path.exists(v_path):
        cds.retrieve("derived-era5-single-levels-daily-statistics",
                     {**req_base, "variable": ["10m_v_component_of_wind"]}, v_path)
        
    if not os.path.exists(ps_path):
        cds.retrieve("derived-era5-single-levels-daily-statistics",
                 {**req_base, "variable": ["surface_pressure"]}, ps_path)
    
    if os.path.exists(u_path) and os.path.exists(v_path) and os.path.exists(ps_path):
        print(f"⏭️ Usando archivos ERA5 ya descargados para {fecha}.")
    
    ds_u = xr.open_dataset(u_path)
    ds_v = xr.open_dataset(v_path)
    ds_ps = xr.open_dataset(ps_path)
    return xr.merge([ds_u, ds_v, ds_ps])

def procesar_tif(in_tif, out_dir):
    """
    Procesa un archivo GeoTIFF con datos de CH₄ para detectar plumas y estimar Q

    Args:
        in_tif (str): ruta al archivo GeoTIFF
        out_dir (str): carpeta donde guardar resultados y logs
    
    Outputs:
        Guarda resultados de plumas (si hay), y logs de procesamiento o errores
        Aplica filtros de anomalía (ppb) y Q (kg/s)
    """

    try:
        t0 = datetime.datetime.now()
        fn = os.path.basename(in_tif)
        
        partes = fn.split('_')
        f_ini = datetime.date.fromisoformat(partes[-2])
        anio = f_ini.year
        
        # Asegurar que la carpeta del año exista
        os.makedirs(out_dir, exist_ok=True)
        
        # Preparar CSV de salida para registrar Q e incertidumbre
        csv_path = os.path.join(out_dir, f"emisiones_ch4_{anio}.csv")
        nuevo_csv = not os.path.exists(csv_path)
        
        with rasterio.open(in_tif) as src:
            ch4 = src.read(1).astype(float)
            meta = src.meta.copy()
            tf = src.transform
            crs = src.crs
        try:
            with rasterio.open(DEM_PATH) as src_dem:
                dem_alineado = np.empty_like(ch4, dtype=np.float32)
                reproject(
                    source=src_dem.read(1),
                    destination=dem_alineado,
                    src_transform=src_dem.transform,
                    src_crs=src_dem.crs,
                    dst_transform=tf,  # transform del Sentinel
                    dst_crs=crs,       # CRS del Sentinel
                    resampling=Resampling.bilinear
                )
                dem = dem_alineado
        except Exception as e:
            print(f"⚠️ No se pudo interpolar DEM: {e}")
            dem = np.full_like(ch4, np.nan)
        
        print(f"⏳ Procesando {fn}...")
        
        if not pre_filtro_tif(in_tif):
            print(f"⚠️ Descartado por pre-filtro: {os.path.basename(in_tif)}")
            registrar_procesamiento(os.path.basename(in_tif), out_dir, 0)
            return

        f_fin = datetime.date.fromisoformat(partes[-1].replace('.tif',''))
        fechas = [f_ini + datetime.timedelta(days=i) for i in range((f_fin-f_ini).days+1)]
        doy = f_ini.timetuple().tm_yday 

        cy, cx = ch4.shape[0]//2, ch4.shape[1]//2
        U10_series = []
        ps_vals = []
        
        for fecha in fechas:
            ds = obtener_era5_u_v(fecha)
        
            # viento
            Uera, Vera = ds.u10.values, ds.v10.values
            U, V = interpolar_viento_a_grilla(Uera, Vera, ch4.shape)
            U10_series.append(np.sqrt(U[cy, cx]**2 + V[cy, cx]**2))
        
            # presión superficial (Pa)
            if "sp" in ds:
                ps_vals.append(float(ds.sp.mean()))
            elif "surface_pressure" in ds:
                ps_vals.append(float(ds.surface_pressure.mean()))
        
        U10_series = np.array(U10_series)
        ps_mean = np.mean(ps_vals) if ps_vals else 101325.0
        print(f"Presión superficial media de la ventana: {ps_mean:.0f} Pa")

        U_c = np.mean(U[cy-3:cy+4, cx-3:cx+4])
        V_c = np.mean(V[cy-3:cy+4, cx-3:cx+4])
        theta_viento = np.arctan2(V_c, U_c)

        rows, cols = np.indices(ch4.shape)
        xs, ys = tf * (cols, rows)
        lon_src, lat_src = rio_transform(crs, 'EPSG:4326', xs.flatten(), ys.flatten())
        lon_src = np.array(lon_src).reshape(ch4.shape)
        lat_src = np.array(lat_src).reshape(ch4.shape)
        
        # Calcular área promedio de píxel en km² (EPSG:4326)
        pix_w_deg = abs(tf.a)
        pix_h_deg = abs(tf.e)
        lat_mean = np.nanmean(lat_src)
        deg2m_lat = 110574.0
        deg2m_lon = 111320.0 * np.cos(np.deg2rad(lat_mean))
        pixel_area_m2 = (pix_w_deg * deg2m_lon) * (pix_h_deg * deg2m_lat)
        pixel_area_km2 = pixel_area_m2 / 1e6
        
        # Definir área mínima física y píxeles equivalentes
        MIN_AREA_KM2 = 200.0   # área mínima en km²
        MIN_AREA_PX = max(3, int(np.ceil(MIN_AREA_KM2 / pixel_area_km2)))
        print(f"Área promedio por píxel: {pixel_area_km2:.1f} km², MIN_AREA_PX={MIN_AREA_PX}")

        # Aplanar variables
        lat_flat = lat_src.ravel()
        lon_flat = lon_src.ravel()
        dem_flat = dem.ravel()
        u10_flat = U.ravel()
        v10_flat = V.ravel()
        ch4_flat = ch4.ravel()

        valid = ~np.isnan(ch4_flat) & ~np.isnan(dem_flat) & ~np.isnan(u10_flat) & ~np.isnan(v10_flat)
        
        if np.sum(valid) < 100:
            print("⚠️ Muy pocos datos válidos para regresión, usando percentil como fondo.")
            fondo0 = np.nanpercentile(ch4, AREA_FRAC_MIN*100)
            anom = ch4 - fondo0
            residuals = None
        else:
            X = np.column_stack([
                lat_flat[valid],
                lon_flat[valid],
                dem_flat[valid],
                u10_flat[valid],
                v10_flat[valid],
                np.full(np.sum(valid), doy)
            ])
            y = ch4_flat[valid]
            modelo = LinearRegression()
            modelo.fit(X, y)
            residuals = y - modelo.predict(X)

            # Predecir fondo
            X_pred = np.column_stack([
                lat_src.ravel(),
                lon_src.ravel(),
                dem.ravel(),
                U.ravel(),
                V.ravel(),
                np.full(lat_src.size, doy)
            ])
            fondo = np.full_like(ch4.ravel(), np.nan)
            pred_valid = ~np.isnan(dem.ravel()) & ~np.isnan(U.ravel()) & ~np.isnan(V.ravel())
            fondo[pred_valid] = modelo.predict(X_pred[pred_valid])
            fondo = fondo.reshape(ch4.shape)
            anom = ch4 - fondo
        
            if np.isnan(anom).all():
                print("⚠️ Fondo regresión dio todo NaN, usando percentil como fallback.")
                fondo0 = np.nanpercentile(ch4, AREA_FRAC_MIN*100)
                anom = ch4 - fondo0
                residuals = None  

        if np.nanmax(anom) < ANOM_THRESHOLD:
            print("  -> ⚠️ Descartado: anomalía máxima insuficiente")
            registrar_procesamiento(fn, out_dir, 0)
            return

        dlat = lat_src - lat_src[cy, cx]
        dlon = lon_src - lon_src[cy, cx]
        theta_pix = np.arctan2(dlat, dlon)
        delta_ang = np.abs((theta_pix - theta_viento + np.pi) % (2 * np.pi) - np.pi)
        upwind = delta_ang < (np.pi / 2)
        bg_vals = anom[upwind & ~np.isnan(anom)].ravel()
        print(f"  -> T-test: {len(bg_vals)} píxeles barlovento usados")

        mask = crear_mascara_pluma(anom, bg_vals)
        print(f"  -> Tamaño de máscara inicial (pre-filtro): {np.sum(mask)} píxeles")
        if mask.sum() < MIN_AREA_PX:
            print(f"  -> ⚠️ Descartado: máscara muy pequeña ({mask.sum()} píxeles < {MIN_AREA_PX})")
            registrar_procesamiento(fn, out_dir, 0)
            return


        etiquetas = label(mask)
        regiones = regionprops(etiquetas)
        print(f"  -> Regiones detectadas antes del filtro de elongación: {len(regiones)}")
        validas = []
        for reg in regiones:
            if reg.area < MIN_AREA_PX:
                continue
            submask = (etiquetas==reg.label)
            if not es_pluma_alargada(submask):
                continue
            validas.append(submask)
        print(f"  -> Regiones válidas después del filtro: {len(validas)}")
        if not validas:
            print("  -> ⚠️ Descartado: no hay plumas válidas")
            registrar_procesamiento(fn, out_dir, 0)
            return

        lat0 = lat_src[cy, cx]
        # Convertir IME de ppb·m² a kg 
        # constantes físicas
        G = 9.80665          # m/s2
        M_AIR = 0.0289647    # kg/mol
        M_CH4 = 0.01604      # kg/mol
        
        # Masas de aire por m2 (mol/m2) - aproximación con presión superficial
        moles_air_m2 = ps_mean / (G * M_AIR)   # mol/m2 (scalar)
        
        anom_masked = np.where(mask, anom, 0.0)   # ppb
        
        # convertir a kg/m2 por píxel
        kg_per_m2 = anom_masked * 1e-9 * moles_air_m2 * M_CH4   # kg/m2
        
        # masa por píxel y suma total -> IME en kg
        mass_per_pixel = kg_per_m2 * pixel_area_m2   # kg (2D)
        IME_kg = np.nansum(mass_per_pixel)
        
        print(f"    -> IME convertido: {IME_kg:.3e} kg (ps_mean={ps_mean:.0f} Pa, moles_air_m2={moles_air_m2:.3e})")

        y_idx, x_idx = np.where(mask)
        if len(y_idx) > 0:
            dist = np.sqrt((lat_src[y_idx, x_idx] - lat0)**2 + (lon_src[y_idx, x_idx] - lon_src[cy, cx])**2) * 111000
            L = dist.max()
        else:
            L = 0

        # Calcular Q con IME_kg 
        Q, Ueff, tau = calcular_Q(IME_kg, L, U10_series, alpha1=1.0, alpha2=0.6, max_iter=10)
        
        print(f"    -> Resultado: IME_kg={IME_kg:.3e} kg, tau={tau:.1f} s, Ueff={Ueff:.2f} m/s, Q={Q:.3f} kg/s")

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if nuevo_csv:
                writer.writerow([
                    'fecha_inicio', 'fecha_fin', 'pluma_id', 'Q_kgps',
                    'sigma_rel', 'sigma_abs',
                    'sigma_rel_mc', 'sigma_abs_mc',
                    'Q_low_95', 'Q_high_95','IQR','IQR_rel'
                ])
        
            for idx, reg in enumerate(validas, 1):
                num_px = np.sum(reg)
                print(f"\n🔍 Evaluando pluma {idx} con {num_px} píxeles...")
            
                # --- Fracción dentro de Colombia ---
                ys_i, xs_i = np.where(reg)
                lon_pluma = lon_src[ys_i.min():ys_i.max()+1, xs_i.min():xs_i.max()+1]
                # CORRECCIÓN: segunda dimensión debe usar xs_i (antes usabas ys_i dos veces)
                lat_pluma = lat_src[ys_i.min():ys_i.max()+1, xs_i.min():xs_i.max()+1]
                geom = box(lon_pluma.min(), lat_pluma.min(), lon_pluma.max(), lat_pluma.max())
                geom_proj = transform(project, geom)
                inter = geom_proj.intersection(geom_colombia_proj)
                frac_dentro = inter.area / geom_proj.area if geom_proj.area > 0 else 0
            
                if frac_dentro < 0.7:
                    print(f"⚠️ Pluma {idx} descartada: solo {frac_dentro:.1%} dentro de Colombia")
                    continue
            
                # --- ANOMALÍA y IME en kg (por pluma) ---
                # Extraemos solo los píxeles de la pluma: evita mezclar con la anomalia global
                anom_px = anom[reg]                # 1D array de anomalías (ppb) solo de la pluma
                # opcional: usar solo anomalías positivas (como haces en otros lugares)
                pos_anom_px = np.where(anom_px > 0, anom_px, 0.0)
            
                # calcular kg/m2 por píxel y masa por píxel (local)
                conv_factor = (ps_mean / (G * M_AIR)) * M_CH4 * 1e-9
                # area por píxel en la huella; construimos vector de tamaño igual a pos_anom_px
                area_px_mask = np.full(pos_anom_px.shape, pixel_area_m2, dtype=float)
            
                mass_per_pixel = pos_anom_px * conv_factor * area_px_mask  # kg por píxel
                ime_i = float(np.nansum(mass_per_pixel))
            
                if ime_i <= 0:
                    print(f"⚠️ Pluma {idx} descartada: IME={ime_i:.1f} kg es negativo o cero")
                    registrar_procesamiento(fn, out_dir, 0)
                    return
            
                # --- Longitud de pluma (en m) ---
                lat0_i = lat_src[cy, cx]
                dist_i = np.sqrt(
                    (lat_src[ys_i, xs_i] - lat0_i) ** 2 +
                    (lon_src[ys_i, xs_i] - lon_src[cy, cx]) ** 2
                ) * 111000
                L_i = float(dist_i.max()) if len(dist_i) > 0 else 0.0
            
                # --- Calcular Q punto estimado ---
                Q_i, Ueff_i, tau_i = calcular_Q(ime_i, L_i, U10_series)
            
                if Q_i < Q_THRESHOLD:
                    print(f"⚠️ Pluma {idx} descartada: Q={Q_i:.3f} kg/s < {Q_THRESHOLD} kg/s")
                    continue
            
                if num_px < MIN_AREA_PX:
                    print(f"⚠️ Pluma {idx} descartada: área {num_px} < MIN_AREA_PX={MIN_AREA_PX}")
                    continue
            
                if not es_pluma_alargada(reg):
                    print(f"⚠️ Pluma {idx} descartada: elongación insuficiente")
                    continue
            
                # --- Pluma aceptada (punto) ---
                print(f"✅ Pluma {idx} aceptada: IME={ime_i:.2e} kg, Q={Q_i:.2f} kg/s, tau={tau_i:.1f} s, Ueff={Ueff_i:.2f} m/s")
            
                # Guardar GeoTIFF de Q 
                win = Window(xs_i.min(), ys_i.min(), xs_i.max()-xs_i.min()+1, ys_i.max()-ys_i.min()+1)
                meta_i = meta.copy()
                meta_i.update({"height": win.height, "width": win.width, "transform": tf * tf.translation(xs_i.min(), ys_i.min())})
                out_tif = os.path.join(out_dir, fn.replace('.tif', f'_pluma{idx}.tif'))
                pluma_q = np.zeros_like(reg, dtype=np.float32)
                pluma_q[reg] = Q_i
                with rasterio.open(out_tif, 'w', **meta_i) as dst:
                    dst.write(pluma_q[ys_i.min():ys_i.max()+1, xs_i.min():xs_i.max()+1], 1)
            
            
                # Validación Monte Carlo
                print("[MC] arrancando Monte Carlo (fusionado corregido)")
            
                Nmc = 5000
                rng = np.random.default_rng(42)
                Q_samples = []
            
                # errores (literatura)
                BIAS_CONC   = 0.003
                SIGMA_CONC  = 0.007
                SIGMA_BG    = float(np.nanstd(residuals)) if (residuals is not None) else 0.0  # ppb
                SES_GO      = 0.5
                SIGMA_WIND  = 1.0
                U10_MIN     = 0.55
                U10_MAX     = 15.0
                L_MIN       = 1e3
            
                # robustez numérica (ajusta si quieres)
                REL_L_MIN_FACTOR = 0.8
                REL_L_MAX_FACTOR = 1.25
                ERR_L_base = 0.20
                sigma_ln_L = np.sqrt(np.log(1.0 + (ERR_L_base ** 2)))
            
                # Pre-cálculos locales
                # a) vector de anomalías positivas en ppb: pos_anom_px
                # b) area_px_mask ya definida arriba
                # c) conv_factor ya definido arriba
            
                # pequeño cap para SIGMA_BG: no permitir que el residual sea disparatado respecto a la señal
                mean_pos_anom = np.nanmean(pos_anom_px) if pos_anom_px.size>0 else 0.0
                SIGMA_BG_ppb = min(SIGMA_BG, max(1.0, 0.5 * mean_pos_anom)) if mean_pos_anom>0 else SIGMA_BG
            
                count_L_below = 0
                count_L_above = 0
            
                for i_mc in range(Nmc):
                    # 1) perturbamos la ANOMALÍA por píxel (multiplicativo + ruido fondo acotado)
                    rel_noise = rng.normal(loc=1.0 + BIAS_CONC, scale=SIGMA_CONC, size=pos_anom_px.shape)
                    anom_mc_px = pos_anom_px * rel_noise
                    if SIGMA_BG_ppb > 0:
                        anom_mc_px = anom_mc_px + rng.normal(0.0, SIGMA_BG_ppb, size=anom_mc_px.shape)
                    anom_mc_px = np.where(anom_mc_px > 0, anom_mc_px, 0.0)  # truncar negativos
            
                    # IME perturvado (kg)
                    ime_mc = float(np.nansum(anom_mc_px * area_px_mask) * conv_factor)
                    ime_mc = max(1e-9, ime_mc)
            
                    # 2) L: log-normal relativo + factor extra relativo (1-40%) multiplicativo
                    L_factor = float(np.exp(rng.normal(loc=0.0, scale=sigma_ln_L)))
                    L_factor = np.clip(L_factor, REL_L_MIN_FACTOR, REL_L_MAX_FACTOR)
                    err_L_rel = float(rng.uniform(0.01, 0.40))
                    L_mc = max(L_i * L_factor * (1.0 + err_L_rel), L_MIN)
            
                    if L_factor <= REL_L_MIN_FACTOR: count_L_below += 1
                    if L_factor >= REL_L_MAX_FACTOR: count_L_above += 1
            
                    # 3) viento: sesgo + ruido por paso
                    if np.size(U10_series) > 1:
                        delta_w = rng.normal(0.0, SIGMA_WIND, size=U10_series.shape)
                        f_wind_rel = float(rng.normal(1.0, 0.20))
                        U10_mc = (U10_series * f_wind_rel) + SES_GO + delta_w
                    else:
                        delta_w = rng.normal(0.0, SIGMA_WIND)
                        f_wind_rel = float(rng.normal(1.0, 0.20))
                        U10_mc = np.atleast_1d((U10_series * f_wind_rel) + SES_GO + delta_w)
            
                    U10_mc = np.clip(U10_mc, U10_MIN, U10_MAX)
                    # cap relativo (38%)
                    mean_ref = np.mean(U10_series)
                    mean_mc = np.mean(U10_mc)
                    rel_err = (mean_mc - mean_ref) / (mean_ref + 1e-12)
                    if abs(rel_err) > 0.38:
                        U10_mc = U10_series * (1.0 + np.sign(rel_err) * 0.38)
                    # 4) calcular Q con fallback
                    try:
                        Q_mc, _, tau_mc = calcular_Q(ime_mc, L_mc, U10_mc)
                        if not (np.isfinite(Q_mc) and tau_mc > 0):
                            raise ValueError("Salida no física")
                    except Exception:
                        Ueff0 = calcular_ueff(float(np.mean(U10_mc)), alpha1=1.0, alpha2=0.6)
                        tau0 = max(L_mc / (Ueff0 + 1e-12), 1.0)
                        Q_mc = ime_mc / tau0
            
                    if np.isfinite(Q_mc) and Q_mc > 0:
                        Q_samples.append(float(Q_mc))
                        if i_mc < 6:
                            print(f"[MC {i_mc}] IME={ime_mc:.2e} kg, L={L_mc/1e3:.1f} km, Umean={np.mean(U10_mc):.2f} m/s → Q={Q_mc:.3f} kg/s")
            
                # diagnósticos y trimming
                print(f"[MC diag] fraction L at min-factor: {count_L_below}/{Nmc}, at max-factor: {count_L_above}/{Nmc}")
            
                Q_samples = np.array(Q_samples)
                if Q_samples.size == 0:
                    raise RuntimeError("No hubo muestras válidas en MC")
            
                low_p, high_p = np.nanpercentile(Q_samples, [1, 99])
                keep_mask = (Q_samples >= low_p) & (Q_samples <= high_p)
                num_removed = Q_samples.size - np.count_nonzero(keep_mask)
                Q_trim = Q_samples[keep_mask]
            
                sigma_abs_mc = float(np.std(Q_trim, ddof=1)) if Q_trim.size > 1 else 0.0
                Q_ref = float(Q_i) if (Q_i is not None and np.isfinite(Q_i) and Q_i > 0) else float(np.median(Q_trim))
                sigma_rel_mc = sigma_abs_mc / (Q_ref + 1e-12)
            
                q_low, q_high = np.percentile(Q_trim, [2.5, 97.5]) if Q_trim.size>0 else (np.nan, np.nan)
                q25, q75 = np.percentile(Q_trim, [25, 75]) if Q_trim.size>0 else (np.nan, np.nan)
                iqr_mc = q75 - q25 if (np.isfinite(q25) and np.isfinite(q75)) else np.nan
                iqr_rel_mc = iqr_mc / (Q_ref + 1e-12) if (Q_ref > 0 and np.isfinite(Q_ref)) else np.nan
            
                print(f"Monte Carlo → Q_ref={Q_ref:.3f}, σ_abs={sigma_abs_mc:.3f} kg/s, σ_rel={sigma_rel_mc:.2%}, removed_outliers={num_removed}/{Q_samples.size}")
                
                # ---------------- Propagación directa (analítica) con cap viento 38% ----------------
                # conc_px: valores absolutos XCH4 (ppb) dentro de la pluma -> ya definido como ch4[reg]
                conc_px = ch4[reg] if 'ch4' in globals() else np.array([])
                
                # 1) Error relativo IME = instrumental Sentinel (bias+disp) + residual de fondo
                err_instr_rel = np.sqrt(0.007**2 + 0.003**2)  # 0.7% disp + 0.3% bias
                sigma_bg = np.nanstd(residuals) if ('residuals' in locals() and residuals is not None) else 0.0
                err_bg_rel = (sigma_bg / (np.nanmean(conc_px) + 1e-12)) if conc_px.size > 0 else 0.0
                # opcional: limitar err_bg_rel (ej. nunca > 0.5)
                err_bg_rel = min(err_bg_rel, 0.50)
                err_IME_rel = np.sqrt(err_instr_rel**2 + err_bg_rel**2)
                
                # 2) Error relativo viento: bias 0.5 m/s + ruido 1.0 m/s, convertido a relativo, con CAP = 0.38
                u_mean = np.nanmean(U10_series) if 'U10_series' in globals() else 0.0
                if u_mean > 0:
                    err_wind_rel_raw = np.sqrt((0.5 / u_mean)**2 + (1.0 / u_mean)**2)
                else:
                    err_wind_rel_raw = 1.0  # caso extremo
                
                ERR_WIND_CAP = 0.38 # cap relativo (38%)
                err_wind_rel = min(err_wind_rel_raw, ERR_WIND_CAP)
                
                # 3) Error relativo L adaptativo según ruido (noise_frac = err_bg_rel)
                noise_frac = err_bg_rel
                if noise_frac < 0.05:
                    err_L_rel = 0.10
                elif noise_frac < 0.15:
                    err_L_rel = 0.20
                else:
                    err_L_rel = 0.40
                
                # 4) Propagación combinada (asumiendo independencia)
                sigma_rel_prop = np.sqrt(err_IME_rel**2 + err_wind_rel**2 + err_L_rel**2)
                sigma_abs_prop = Q_i * sigma_rel_prop
                
                # Print comparador claro
                print(f"[PROP] Q_i={Q_i:.3f} kg/s → σ_abs={sigma_abs_prop:.3f} kg/s, σ_rel={sigma_rel_prop:.2%} "
                      f"(IME={err_IME_rel:.2%}, wind={err_wind_rel:.2%} (cap {ERR_WIND_CAP:.0%}), L={err_L_rel:.2%}, noise_frac={noise_frac:.2%})")
                
                # 5) Guardar en CSV 
                writer.writerow([
                   f_ini.isoformat(), f_fin.isoformat(), idx, Q_i,
                   sigma_rel_prop, sigma_abs_prop,
                   sigma_rel_mc, sigma_abs_mc,
                   q_low, q_high,
                   iqr_mc, iqr_rel_mc
               ])

                
        print(f"  -> Tiempo de procesamiento: {(datetime.datetime.now() - t0).total_seconds():.1f} s")
    
        # Registrar que fue procesado
        registrar_procesamiento(fn, out_dir, len(validas))

    except Exception as e:
        print(f"🛑 Error procesando {in_tif}: {e}")
        with open(os.path.join(out_dir, "errores.log"), "a") as log:
            log.write(f"{datetime.datetime.now()}: {in_tif} - {e}\n")
            traceback.print_exc(file=log)

def procesar_ime(año=None):
    #año = 2024
    año = año or int(sys.argv[1]) if len(sys.argv) > 1 else datetime.date.today().year
    in_dir = os.path.join(BASE_SENTINEL, str(año))
    out_dir = os.path.join(BASE_RESULT,   str(año))

    # Cargar lista de procesados
    registro_path = os.path.join(out_dir, "procesados.txt")
    ya_procesados = set()
    if os.path.exists(registro_path):
        with open(registro_path, "r") as f:
            ya_procesados = {line.strip().split(",")[0] for line in f if line.strip()}

    for fn in sorted(os.listdir(in_dir)):
        if not fn.endswith('.tif'): continue
        if fn in ya_procesados:
            print(f"⏭️ {fn} ya está registrado como procesado. Se omite.")
            continue  # Saltar si ya fue procesado
        procesar_tif(os.path.join(in_dir, fn), out_dir)


if __name__ == '__main__':
    procesar_ime(None)
