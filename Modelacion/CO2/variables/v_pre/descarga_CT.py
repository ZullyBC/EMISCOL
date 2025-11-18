# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:57:40 2025

@author: Zully JBC

Nota: Código dispuesto para descargar de forma automática los datos del Carbon Tracker, 
una variable importante para que el modelo ERT estime el fondo del XCO2.
"""
# 1. Importación de librerias
import os
import sys
import re
from ftplib import FTP
from datetime import datetime

# 2. Función para extraer la fecha del nombre de los archivos
def extraer_fecha(filename):
    """
    Extrae la primera ocurrencia de 6 dígitos en el nombre del archivo (por ejemplo, "140906")
    y lo convierte al formato YYYY-MM-DD.
    """
    match = re.search(r'(\d{6})', filename)
    if match:
        fecha_str = match.group(1)
        # Asumimos que años < 50 son del siglo XXI, caso contrario del XX.
        year = int(fecha_str[:2])
        year = year + 2000 if year < 50 else year + 1900
        month = fecha_str[2:4]
        day = fecha_str[4:6]
        return f"{year}-{month}-{day}"
    return None

# 3. Descarga de los datos de Carbon Tracker de NOAA
def descargar_carbon_tracker(oco2_folder, download_folder, anio):
    ftp_host = "aftp.cmdl.noaa.gov"

    rutas_base = [
        "/products/carbontracker/co2/CT2022/molefractions/xCO2_1330LST/",
        "/products/carbontracker/co2/CT-NRT.v2025-1//molefractions/xCO2_1330LST/"
    ]
    nombres_archivo_glb = [
        lambda fecha: f"CT2022.xCO2_1330_glb3x2_{fecha}.nc",
        lambda fecha: f"CT-NRT.v2025-1.xCO2_1330_glb3x2_{fecha}.nc"
    ]

    ftp = FTP(ftp_host)

    try:
        ftp.login()
        print("Conectado al FTP:", ftp_host)

        archivos_oco2 = [f for f in os.listdir(oco2_folder) if f.endswith('.nc4')]

        for archivo_oco2 in archivos_oco2:
            fecha_oco2 = extraer_fecha(archivo_oco2)
            if not fecha_oco2:
                print(f"No se encontró fecha en {archivo_oco2}, lo saltamos.")
                continue
            archivo_descargado = False  

            for i, ruta_base in enumerate(rutas_base):
                ftp.cwd(ruta_base)
                archivos_remotos = ftp.nlst()
                nombre_archivo_glb = nombres_archivo_glb[i](fecha_oco2)
                ruta_local = os.path.join(download_folder, nombre_archivo_glb)

                if os.path.exists(ruta_local):
                    print(f"El archivo {nombre_archivo_glb} ya existe localmente. Omitiendo.")
                    archivo_descargado = True
                    break

                if nombre_archivo_glb in archivos_remotos:
                    print(f"Descargando {nombre_archivo_glb} desde {ruta_base} para OCO-2 {archivo_oco2} ...")
                    with open(ruta_local, 'wb') as f_local:
                        try:
                            ftp.retrbinary("RETR " + nombre_archivo_glb, f_local.write)
                            print(f"Descarga de {nombre_archivo_glb} completada.")
                            break  # Si se descarga de una ruta, no intentar la siguiente
                        except Exception as e:
                            print(f"Error al descargar {nombre_archivo_glb} desde {ruta_base}: {e}")

                if not archivo_descargado:
                    print(f"No se encontró un archivo de Carbon Tracker válido para la fecha {fecha_oco2} correspondiente a {archivo_oco2} en las rutas FTP.")

    except Exception as e:
        print(f"Ocurrió un error general en la conexión FTP: {e}")
    finally:
        if ftp.sock:
            ftp.quit()
            print("Conexión FTP cerrada.")

if __name__ == "__main__":
    #anio = 2024
    anio = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.now().year
    # Obtener directorio del script actual
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Rutas ABSOLUTAS
    carpeta_oco2 = os.path.join(SCRIPT_DIR, "..", "..", "datos_OCO2", "OCO2_L2_Lite_FP_Co")
    carpeta_entrada = os.path.join(carpeta_oco2, str(anio))
    carpeta_descargas = os.path.join(SCRIPT_DIR, "Carbon_Tracker")  # ← Crea en v_pre/Carbon_Tracker/
    carpeta_salida = os.path.join(carpeta_descargas, str(anio))
    os.makedirs(carpeta_salida, exist_ok=True)
    
    descargar_carbon_tracker(carpeta_entrada, carpeta_salida, anio)

