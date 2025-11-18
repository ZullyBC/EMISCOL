# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:40:19 2024

@author: Zully JBC
"""

import sys
import os
import datetime
from datetime import timedelta
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import warnings

# --- Parámetros Copernicus ---
CLIENT_ID     = "client_id"
CLIENT_SECRET = "client_secret"
TOKEN_URL     = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL   = "https://sh.dataspace.copernicus.eu/api/v1/process"

# --- Evalscript: valores crudos de CH4 ---
evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["CH4", "dataMask"],
        output: { bands: 1, sampleType: "FLOAT32" }
    };
}

function evaluatePixel(sample) {
    return [sample.dataMask === 1 ? sample.CH4 : NaN];
}
"""

def get_oauth_session():
    client = BackendApplicationClient(client_id=CLIENT_ID)
    oauth = OAuth2Session(client=client)
    oauth.fetch_token(token_url=TOKEN_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    return oauth

def generar_fechas(year, step_days=5):
    start = datetime.date(year, 1, 1)
    end = datetime.date(year, 12, 31)
    fechas = []
    while start <= end:
        fin = start + timedelta(days=step_days - 1)
        if fin > end:
            fin = end
        fechas.append((start, fin))
        start = fin + timedelta(days=1)
    return fechas

def descargar_y_plot(anio=None, step=5):
    # Obtener directorio del script actual
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    hoy = datetime.date.today()
    year = anio or int(sys.argv[1]) if len(sys.argv) > 1 else hoy.year
    
    # Ruta ABSOLUTA - crea en datos_CH4/SENTINEL_5P_L2/
    base_dir = os.path.join(SCRIPT_DIR, "SENTINEL_5P_L2")            
    output_dir = os.path.join(base_dir, str(year))
    os.makedirs(output_dir, exist_ok=True)

    oauth = get_oauth_session()

    for f_ini, f_fin in generar_fechas(year, step):
        print(f"🔄 {f_ini} → {f_fin}")
        request = {
            "input": {
                "bounds": {
                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                    "bbox": [-79.01, -4.226, -66.857, 12.46]
                },
                "data": [{
                    "type": "sentinel-5p-l2",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{f_ini.isoformat()}T00:00:00Z",
                            "to":   f"{f_fin.isoformat()}T23:59:59Z"
                        }
                    },
                    "processing": {
                        "minQa": 50   # 🔹 filtrado de calidad mínimo recomendado
                    }
                }]
            },
            "evalscript": evalscript,
            "output": {
                "responses": [{
                    "identifier": "default",
                    "format": {"type": "image/tiff"}
                }]
            }
        }

        resp = oauth.post(PROCESS_URL, json=request)
        if resp.status_code != 200:
            print(resp.text)
            warnings.warn(f"❌ Error {resp.status_code} en rango {f_ini}–{f_fin}")
            continue

        tiff_path = os.path.join(output_dir, f"sentinel5p_CH4_{f_ini}_{f_fin}.tif")
        
        if os.path.exists(tiff_path):
            print(f"📁 Ya existe: {tiff_path} — se salta el archivo")
            continue
        
        with open(tiff_path, "wb") as f:
            f.write(resp.content)
        print(f"📦 Guardado: {tiff_path}")

        # --- Verificación y plot ---
        try:
            with rasterio.open(tiff_path) as src:
                ch4_data = src.read(1)

            if np.isnan(ch4_data).all():
                warnings.warn(f"⚠️ Todo NaN en {tiff_path}, se borró.")
                os.remove(tiff_path)
                continue

            plt.figure(figsize=(8, 5))
            plt.imshow(
                ch4_data,
                cmap="viridis",
                vmin=np.nanpercentile(ch4_data, 5),
                vmax=np.nanpercentile(ch4_data, 95)
            )
            plt.colorbar(label="CH₄ [ppb]")
            plt.title(f"CH₄ {f_ini} → {f_fin}")
            plt.xlabel("Pixel X")
            plt.ylabel("Pixel Y")
            plt.show()                         

        except Exception as e:
            warnings.warn(f"💥 Falló al procesar {tiff_path}: {e}")
            if os.path.exists(tiff_path):
                os.remove(tiff_path)


if __name__ == "__main__":
    descargar_y_plot(anio=None, step=5)
