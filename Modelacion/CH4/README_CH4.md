# 💨 Script de Modelación Inversa de CH₄

Esta carpeta contiene el script principal que implementa el **Método IME (Integrated Mass Enhancement)** para el análisis y la cuantificación de las emisiones de Metano ($\text{CH}_4$), utilizando datos satelitales Sentinel-5P.

## ⚙️ Script Incluido

* **`IME.py`**: Script principal que integra y orquesta la descarga de datos de y ERA5, el pre-procesamiento, la aplicación del Método IME, la detección de plumas y el cálculo del flujo de emisión de $\text{CH}_4$.
* `sentinel5p_descarga.py`: Script de utilidad para la obtención de los datos de Sentinel-5P.

## 📥 Datos de Entrada Requeridos

Para la ejecución de `IME.py`, el código buscará los siguientes archivos y carpetas, que deben estar poblados en la estructura de carpetas definida en el repositorio raíz:

| Tipo de Dato | Ubicación (Relativa a la Raíz) | Ejemplo de Uso |
| :--- | :--- | :--- |
| **Satélite ($\text{CH}_4$)** | `/datos_CH4/SENTINEL_5P_L2/` | Archivos NetCDF de Sentinel-5P. |
| **Variables ERA5** | `/variables/ERA5/` | Componentes del viento y presión superficial. |
| **Geometría** | `/variables/shp/limite_colombia.shp` | Usado para definir el área de estudio. |
| **DEM** | `/variables/SRTM/SRTM_Colombia.tif` | Modelo de Elevación Digital. |

### 🛠️ Ejecución

Asegúrese de ejecutar el script principal desde la **raíz del repositorio (`/EMISCOL`)** para que las rutas relativas funcionen correctamente.

```bash
python Modelacion/CH4/IME.py
