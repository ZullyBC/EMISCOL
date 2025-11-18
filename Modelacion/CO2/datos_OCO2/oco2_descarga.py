# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:28:20 2024

Nota: La finalidad de este código es ingresar a la plataforma de la NASA
Earthdata y obtener así los datos dados por su satélite OCO-2. Se toman los datos
de OCO-2 L2 Lite, que son más livianos que el L2 Estándar, sin embargo contienen 
las correcciones de sesgo.

@author: Zully JBC
"""

# Importación de librerías
import os
import sys
import requests
from bs4 import BeautifulSoup
from netrc import netrc
from datetime import datetime

# URL base para los datos de OCO-2 de la NASA
BASE_URL = 'https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L2_Lite_FP.11.2r/'

# Carpeta para guardar los archivos descargados
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Obtener directorio del script actual
CARPETA_LOCAL = os.path.join(SCRIPT_DIR,"OCO2_L2_Lite_FP") 

def crear_sesion_autenticada():
    """
    Lee las credenciales desde ~/.netrc y crea una sesión requests con auth.
    """
    host = 'urs.earthdata.nasa.gov'
    creds = netrc().authenticators(host)
    if creds is None:
        raise ValueError(f"No hay autenticadores para {host} en ~/.netrc")
    username, _, password = creds

    session = requests.Session()
    session.auth = (username, password)
    return session

def obtener_archivos_nc4(url, session):
    """
    Devuelve lista de nombres *.nc4 en la URL dada.
    """
    resp = session.get(url)
    if resp.status_code != 200:
        return []  # no existe esa carpeta de año
    soup = BeautifulSoup(resp.text, 'html.parser')
    return [a['href'] for a in soup.find_all('a') if a['href'].endswith('.nc4')]

def descargar_archivo(url, carpeta_destino, session):
    """
    Descarga el archivo si no existe localmente.
    """
    nombre = url.split('/')[-1]
    ruta = os.path.join(carpeta_destino, nombre)

    if not os.path.exists(ruta):
        print(f"⬇️ Descargando {nombre}…")
        resp = session.get(url, stream=True)
        resp.raise_for_status()
        with open(ruta, 'wb') as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        print(f"✅ Guardado en {ruta}")
        return True
    return False

def verificar_y_descargar_archivos_de_anio(anio, session):
    """
    Busca y descarga todos los .nc4 de un año dado.
    Devuelve el número de archivos nuevos descargados.
    """
    url_anio = f"{BASE_URL}{anio}/"
    archivos = obtener_archivos_nc4(url_anio, session)

    if not archivos:
        return 0

    carpeta_anio = os.path.join(CARPETA_LOCAL, str(anio))
    os.makedirs(carpeta_anio, exist_ok=True)

    nuevos = 0
    for nc4 in archivos:
        full_url = url_anio + nc4
        if descargar_archivo(full_url, carpeta_anio, session):
            nuevos += 1

    if nuevos == 0:
        print(f"⚠️  Ningún archivo nuevo para {anio}.")
    return nuevos

if __name__ == "__main__":
    session = crear_sesion_autenticada()

    # 1. Si no se provee año, usar el actual
    anio = None
    if anio is None:
        anio = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.now().year  # año actual dinámico :contentReference[oaicite:1]{index=1}

    # 2. Intentar descargar; si no hay datos, ir restando años
    while anio >= 2020:  # límite mínimo: primer año disponible
        print(f"\n>>> Intentando año {anio}...")
        nuevos = verificar_y_descargar_archivos_de_anio(anio, session)
        if nuevos > 0:
            print(f"✅ Se descargaron {nuevos} archivos para el año {anio}.")
            break
        else:
            print(f"🔄 No hay datos en {anio}, probando {anio - 1}...")
            anio -= 1
    else:
        print("❌ No se encontraron datos en ningún año desde 2014 hasta ahora.")

