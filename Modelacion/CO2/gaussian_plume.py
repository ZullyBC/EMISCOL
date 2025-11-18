# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:03:09 2025

@author: Zully JB

Nota: En este script se manejan las funciones creadas necesarias para la modelación
inversa de la pluma gaussina, con el fin de detectar los puntos de emisión CO2. 
"""

import os
import numpy as np
import xarray as xr
import alphashape
from affine import Affine
from scipy.ndimage import label, generate_binary_structure, uniform_filter, median_filter
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
from shapely.validation import make_valid


def calcular_baseline_grid(co2_grid, window_size=7, n_sigma=2, min_cluster_size=5):
    """
     Calcula el valor de referencia (baseline) y detecta anomalías significativas en la grilla de CO2.
    
     Parámetros:
       co2_grid (ndarray): Matriz de concentraciones de CO2.
       window_size (int): Tamaño de la ventana para el filtro de mediana y cálculo de desviación.
       n_sigma (float): Umbral de número de desviaciones estándar para considerar anomalía.
       min_cluster_size (int): Tamaño mínimo de cluster para considerar válida la anomalía.
    
     Retorna:
       baseline (ndarray): Matriz filtrada con mediana local.
       final_mask (ndarray bool): Mascara binaria con clusters de anomalías.
     """
    baseline = median_filter(co2_grid, size=window_size) # Aplica filtro de mediana para estimar baseline local
    delta = np.nan_to_num(co2_grid - baseline, nan=0.0) # Calcula la diferencia entre valores reales y baseline
    delta_sq = np.nan_to_num(delta**2, nan=0.0, posinf=0.0, neginf=0.0)
    std_local = np.sqrt(np.maximum(uniform_filter(delta_sq, size=window_size), 0)) # Estima desviación estándar local mediante filtro unifor
    z_scores = (co2_grid - baseline) / (std_local + 1e-6) # Calcula puntuaciones z para cada punto
    mask = z_scores > n_sigma # Genera máscara inicial de valores por encima de n_sigma
    estructura = generate_binary_structure(2, 2) # Define estructura de conectividad para clusters 2D
    labeled, num = label(mask, structure=estructura) # Etiqueta clusters conectados
    final_mask = np.zeros_like(mask, dtype=bool)
    
    # Conserva solo clusters con tamaño >= min_cluster_size
    for lab in range(1, num+1):
        cluster = (labeled == lab)
        if cluster.sum() >= min_cluster_size:
            final_mask |= cluster
    return baseline, final_mask


def detectar_areas_emision(anomalias, lat_grid, lon_grid, eps=0.2, min_samples=7, alpha=0.05):
    """
    Agrupa píxeles anómalos en regiones contiguas y genera polígonos alfa que representan áreas de emisión.
    
    Parámetros:
      anomalías (ndarray bool): Mascara de anomalías en la grilla.
      lat_grid, lon_grid (ndarray): Coordenadas geográficas de cada punto.
      eps (float): Distancia máxima (grados) para vecinos en DBSCAN.
      min_samples (int): Número mínimo de puntos para formar un cluster.
      alpha (float): Parámetro alfa para la construcción de la envolvente alfa.
    
    Retorna:
      areas (list[Polygon]): Lista de polígonos válidos de emisión.
    """
    ys, xs = np.where(anomalias) # Extrae índices de puntos anomalos
    pts = [(lon_grid[y,x], lat_grid[y,x]) for y,x in zip(ys,xs)] # Construye lista de coordenadas (lon, lat)
    # Aplica DBSCAN para identificar clusters de anomalías
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = db.labels_
    areas = []
    
    # Itera sobre cada etiqueta de cluster
    for lab in set(labels):
        if lab < 0: continue # descarta ruido
        cluster_pts = [pts[i] for i in range(len(pts)) if labels[i]==lab]
        if len(cluster_pts) < min_samples: continue  # descarta clusters pequeños
        coords = np.array(cluster_pts)
        try:
            poly = alphashape.alphashape(coords, alpha) if len(coords)>=4 else MultiPoint(coords).convex_hull
            poly = make_valid(poly).buffer(0.5) # Asegura geometría válida y aplica buffer mínimo
            if not poly.is_empty:
                areas.append(poly)
        except:
            continue
    return areas


def parametros_dispersion(x, viento_10m, nubosidad=None):
    """
    Calcula sigma_y y sigma_z usando coeficientes de Espert (2000).

    Parámetros:
      x (ndarray): Distancias en metros
      viento_10m (float): Velocidad del viento a 10m
      nubosidad (float, opcional): Fracción de nubosidad (0 a 1)

    Retorna:
      sigma_y, sigma_z (ndarray): Dispersividades horizontales y verticales
    """

    # Clasificación según viento y nubosidad
    def clasificar_estabilidad(viento, nub):
        if viento >= 5:
            return 'D'
        elif nub is not None:
            if viento < 2:
                return 'F' if nub < 0.5 else 'E'
            elif viento < 3:
                return 'E' if nub < 0.5 else 'D'
            else:
                return 'D'
        else:
            if viento < 2:
                return 'F'
            elif viento < 3:
                return 'E'
            else:
                return 'D'

    clase = clasificar_estabilidad(viento_10m, nubosidad)

    # Coeficientes según Espert (2000)
    coeficientes = {
        'A': {'a': 0.527, 'b': 0.865, 'c': 0.280, 'd': 0.90},
        'B': {'a': 0.371, 'b': 0.866, 'c': 0.230, 'd': 0.85},
        'C': {'a': 0.209, 'b': 0.897, 'c': 0.220, 'd': 0.80},
        'D': {'a': 0.128, 'b': 0.905, 'c': 0.200, 'd': 0.76},
        'E': {'a': 0.098, 'b': 0.902, 'c': 0.150, 'd': 0.73},
        'F': {'a': 0.065, 'b': 0.902, 'c': 0.120, 'd': 0.67}
    }

    c = coeficientes[clase]
    sigma_y = np.clip(c['a'] * (x ** c['b']), 50, 5000)
    sigma_z = np.clip(c['c'] * (x ** c['d']), 20, 2000)

    return sigma_y, sigma_z



def modelo_pluma_gaussiana_inversa(Q, u, sigma_y, sigma_z, x_rot, y_rot, H=10.0, max_distance_km=100.0):
    """
    Modelo gaussiano inverso de pluma puntual para estimar concentración de masa C.
    
    Parámetros:
      Q (float): Emisión puntual (kg/s).
      u (ndarray): Velocidad del viento en origen (m/s).
      sigma_y, sigma_z (ndarray): Parámetros de dispersión (m).
      x_rot, y_rot (ndarray): Coordenadas rotadas respecto a la dirección del viento (m).
      H (float): Altura de liberación (m).
      max_distance_km (float): Distancia máxima a considerar (km).
    
    Retorna:
      C_mass (ndarray): Concentración de masa en cada celda.
    """
    # Calcula distribución gaussiana
    C_mass = (Q / (2 * np.pi * u * sigma_y * sigma_z)) \
             * np.exp(-y_rot**2/(2*sigma_y**2)) \
             * (np.exp(-H**2/(2*sigma_z**2)))
    # Aplica máscara de rango para descartar resultados fuera de distancia
    mask = (x_rot>0)&(x_rot<=max_distance_km*1000)
    C_mass[~mask] = 0
    return C_mass


def interpolar_era5(era5_ds, lat_grid, lon_grid, variable):
    """
     Interpola una variable de ERA5 sobre la grilla de lat-lon usando interpolación lineal y nearest.
    
     Parámetros:
       era5_ds (xarray.Dataset): Conjunto de datos ERA5 con coords 'lat' y 'lon'.
       lat_grid, lon_grid (ndarray): Grillas destino de interpolación.
       variable (str): Nombre de la variable a interpolar.
    
     Retorna:
       datos (ndarray): Valores interpolados en la grilla.
     """
    if era5_ds is None:
        return np.zeros_like(lat_grid)
    lat_e = era5_ds.coords['lat'].values
    lon_e = era5_ds.coords['lon'].values
    vals = era5_ds[variable].values.squeeze().flatten()
    pts = np.array([(la,lo) for la in lat_e for lo in lon_e])
    mask_valid = ~np.isnan(vals)
    # Interpolación lineal inicial
    datos = griddata(pts[mask_valid], vals[mask_valid], (lat_grid, lon_grid), method='linear')
    # Rellena restantes con nearest si quedan NaNs
    if np.isnan(datos).any():
        datos = griddata(pts[mask_valid], vals[mask_valid], (lat_grid, lon_grid), method='nearest')
    return datos


def construir_matriz_sensibilidad_grid(areas, lat_mesh, lon_mesh, inst_era5, anom_grid,
    buffer_km=50, H=10.0):
    """
    Construye la matriz A (sensibilidad) relacionando emisiones puntuales y respuestas en cada celda.
    
    Parámetros:
      areas (list[np.ndarray bool]): Lista de máscaras binarias de cada área.
      lat_mesh, lon_mesh (ndarray): Malla 2D de coordenadas.
      inst_era5 (xr.Dataset): ERA5 interpolado a la grilla.
      anom_grid (ndarray): Grid de anomalías para localizar hotspots.
      buffer_km (float): Radio de influencia máximo (km).
      H (float): Altura de liberación (m).
      normalize (bool): Si True, normaliza cada columna de A (guardando factor).
    
    Retorna:
      A (ndarray): Matriz (n_obs, n_areas).
      norm_factors (list): Factores de escala aplicados a cada columna.
    """
    n_obs = lat_mesh.size
    n_areas = len(areas)
    A = np.zeros((n_obs, n_areas))
    norm_factors = []

    # Campos de viento
    u = interpolar_era5(inst_era5, lat_mesh, lon_mesh, 'u10')
    v = interpolar_era5(inst_era5, lat_mesh, lon_mesh, 'v10')

    for i, mask in enumerate(areas):
        print(f"Procesando área {i+1}/{n_areas}...")
        ys, xs = np.where(mask)
        if ys.size == 0:
            norm_factors.append(1.0)
            continue

        # 1. Hotspot = punto de mayor anomalía
        ani_vals = anom_grid[mask]
        max_idx = np.nanargmax(ani_vals)
        y0, x0 = ys[max_idx], xs[max_idx]
        lon0, lat0 = lon_mesh[y0, x0], lat_mesh[y0, x0]

        # 2. Viento promedio local (5 km buffer)
        dx = (lon_mesh - lon0) * 111320 * np.cos(np.radians(lat0))
        dy = (lat_mesh - lat0) * 111320
        circle_mask = (dx**2 + dy**2 <= (5000)**2)
        u0 = np.nanmean(u[circle_mask])
        v0 = np.nanmean(v[circle_mask])
        vel0 = np.sqrt(u0**2 + v0**2)
        ang0 = np.arctan2(v0, u0)

        if not np.isfinite(vel0) or vel0 <= 0:
            vel0 = 1.0
            ang0 = 0.0

        # 3. Coordenadas rotadas
        x_rot = dx * np.cos(ang0) + dy * np.sin(ang0)
        y_rot = -dx * np.sin(ang0) + dy * np.cos(ang0)

        # 4. Parámetros de dispersión
        sigma_y, sigma_z = parametros_dispersion(np.abs(x_rot), vel0)

        # 5. Pluma inversa con Q=1
        C_mass = modelo_pluma_gaussiana_inversa(
            Q=1.0,
            u=vel0,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            x_rot=x_rot,
            y_rot=y_rot,
            H=H,
            max_distance_km=buffer_km
        )
        
        # 6. Convertir C a ppm
        # Obtener la densidad del aire (kg_aire/m³) en el punto de la fuente (y0, x0)
        sp0 = inst_era5['sp'].values[y0, x0]  # Presión superficial en Pa
        t2m0 = inst_era5['t2m'].values[y0, x0]  # Temperatura a 2m en K
        
        # Calcular la densidad del aire (usando la ley de los gases ideales)
        R_air = 287.05  # Constante específica del aire en J/kg/K
        rho_air = sp0 / (R_air * t2m0)  # Densidad del aire en kg/m³
        
        # Convertir la concentración de masa (kg_CO2/m³) a ppm
        # ppm = (kg_CO2 / m³) / (kg_air / m³) * (kg_air/mol) / (kg_CO2/mol) * 1e6
        M_air = 0.029  # Masa molar del aire (kg/mol)
        M_co2 = 0.044  # Masa molar del CO2 (kg/mol)
        C_ppm = C_mass * (M_air / M_co2) * (1e6 / rho_air)
        
        # 7. Construir matriz de sensibilidad A
        col = C_ppm.flatten()
        A[:, i] = col
        
    return A



def simular_pluma(x_mesh, y_mesh, lon0, lat0, Q, vel, dir_wind, H=10, buffer_km=50, length_min_km=1.5):
    """
       Simula una pluma gaussiana directa dada posición de fuente y condiciones de viento.
    
       Parámetros:
         x_mesh, y_mesh (ndarray): Grillas de coordenadas (lon, lat).
         lon0, lat0 (float): Coordenadas de la fuente de emisión.
         Q (float): Emisión puntual (kg/s).
         vel (float): Magnitud de la velocidad del viento (m/s).
         dir_wind (float): Dirección del viento en grados.
         H (float): Altura de liberación (m).
         buffer_km (float): Distancia máxima de influencia (km).
    
       Retorna:
         pluma (ndarray): Matriz de concentraciones simuladas.
    """
    
    # Coordenadas en metros desde el origen
    xr = (x_mesh - lon0) * 111000 * np.cos(np.radians(lat0))
    yr = (y_mesh - lat0) * 111000

    # Rotar coordenadas al eje del viento
    theta = np.radians(180 + dir_wind)  # dirección meteorológica estándar
    x_rot = xr * np.cos(theta) - yr * np.sin(theta)
    y_rot = xr * np.sin(theta) + yr * np.cos(theta)

    # Máscara: sólo downwind y dentro de buffer
    mask = (x_rot > 0) & (x_rot <= buffer_km * 1000)
    if not np.any(mask):
        return None

    # Dispersión gaussiana
    sigma_y, sigma_z = parametros_dispersion(np.abs(x_rot), vel)

    pluma = np.zeros_like(x_mesh, dtype=float)
    pluma[mask] = (
        Q / (2 * np.pi * sigma_y[mask] * sigma_z[mask] * vel)
        * np.exp(-y_rot[mask] ** 2 / (2 * sigma_y[mask] ** 2))
        * np.exp(-H ** 2 / (2 * sigma_z[mask] ** 2))
    )

    if np.count_nonzero(pluma) == 0:
        return None

    # Longitud de la pluma
    longitud_km = (x_rot[mask] / 1000).max()
    if longitud_km < length_min_km:
        print(f"⚠️ Pluma descartada: longitud {longitud_km:.2f} km < {length_min_km} km")
        return None

    return pluma


def guardar_pluma(pluma, lat, lon, tag, i, output_dir):
    """
    Guarda la pluma simulada como un archivo GeoTIFF con metadatos CRS y transform.

    Parámetros:
      pluma (ndarray): Matriz de concentración guardada.
      lat, lon (ndarray): Vectores de coordenadas geográficas.
      tag (str): Identificador temporal para el nombre del archivo.
      i (int): Índice del área de emisión.
      output_dir (str): Carpeta destino para almacenar el ráster.

    Retorna:
      None (genera archivo en disco).
    """ 
    # 1. Limpieza de NaNs/infinito
    pluma = np.nan_to_num(pluma, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Recorte al ROI
    ys, xs = np.where(pluma > 0)
    if ys.size:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        lat_crop = lat[y0:y1+1]
        lon_crop = lon[x0:x1+1]
        pluma_crop = pluma[y0:y1+1, x0:x1+1]
    else:
        lat_crop, lon_crop, pluma_crop = lat, lon, pluma

    # ⚡ 3. Ajuste de orientación (flip si las latitudes están invertidas)
    if lat_crop[0] > lat_crop[-1]:
        lat_crop = lat_crop[::-1]
        pluma_crop = np.flipud(pluma_crop)

    # 4. Crear DataArray con coords corregidas
    da = xr.DataArray(
        pluma_crop,
        dims=("latitude", "longitude"),
        coords={"latitude": lat_crop, "longitude": lon_crop},
        name=f"pluma_{i}"
    )

    # 5. CRS y transform
    da = da.rio.write_nodata(0).rio.write_crs("EPSG:4326")
    dx = float(lon_crop[1] - lon_crop[0])
    dy = float(lat_crop[1] - lat_crop[0])
    transform = Affine.translation(lon_crop.min() - dx/2,
                                   lat_crop.min() - dy/2) * Affine.scale(dx, dy)
    da = da.rio.write_transform(transform)

    # 6. Guardar GeoTIFF
    out_plume = os.path.join(output_dir, f"pluma_{tag}_{i}.tif")
    da.rio.to_raster(out_plume, dtype="float32")

    print(f"✅ Pluma guardada y etiquetada: {out_plume}")