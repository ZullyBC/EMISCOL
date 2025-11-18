# 💨 Scripts de Modelación Inversa de CO₂

Esta carpeta contiene los scripts específicos para la estimación de las emisiones de Dióxido de Carbono ($\text{CO}_2$), utilizando el **Modelo Inverso de la Pluma Gaussiano** y datos satelitales OCO-2 junto con variables auxiliares.

## ⚙️ Scripts Incluidos

* `main_CO2.py`: Script principal que orquesta ejecuta la modelación de la pluma gaussiana y post-procesamiento de los resultados de $\text{CO}_2$.
* `gaussian_plume.py`: Módulo con las funciones centrales para construir la matriz de sensibilidad y resolver la inversión del modelo gaussiano.
* `descarga_OCO2_y_vars.py`: Scripts de utilidad para la obtención y organización de los datos de entrada (OCO-2, ERA5, Carbon Tracker, etc.).

## 📥 Datos de Entrada Requeridos

Para la ejecución completa de `main_CO2.py`, el código buscará los siguientes archivos y carpetas, que deben estar poblados con los datos satelitales y variables meteorológicas en la estructura de carpetas definida en el repositorio raíz:

| Tipo de Dato | Ubicación (Relativa a la Raíz) | Ejemplo de Uso |
| :--- | :--- | :--- |
| **Satélite ($\text{CO}_2$)** | `/datos_OCO2/OCO2_L2_Lite_FP_Co/` | Archivos NetCDF de OCO-2. |
| **Variables Metereológicas** | `/variables/` | Variables utilizadas por el modelo de ML. |
| **Modelo Auxiliar** | `/modelos_ML/modelo_ert_xco2.joblib` | Modelo de Machine Learning pre-entrenado. |
| **Geometría** | `/variables/shp/limite_colombia.shp` | Usado para enmascaramiento y recorte. |

### 🛠️ Ejecución y Dependencias del Modelo (Importante)

Asegúrese de ejecutar el script principal desde la **raíz del repositorio (`/EMISCOL`)** para que las rutas relativas funcionen correctamente.

**Dependencia del Modelo de ML:**
Debido a que el archivo binario del modelo de Machine Learning (`modelo_ert_xco2.joblib`) es muy pesado y no puede ser alojado en GitHub, este debe ser **re-entrenado localmente** antes de ejecutar el análisis de $\text{CO}_2$.

**Pasos para la Ejecución:**

1.  **Entrenamiento del Modelo:** Ejecute primero el script `ert_xco2_parameters.py`. Este script leerá los datos necesarios (que deben estar en la carpeta `/variables` con la estructura correcta) y generará el archivo `modelo_ert_xco2.joblib` en la carpeta `/modelos_ML`.
    ```bash
    python ert_xco2_parameters.py
    ```
2.  **Análisis Principal ($\text{CO}_2$):** Una vez que el modelo exista, ejecute el script principal.
    ```bash
    python main_CO2.py
    ```
