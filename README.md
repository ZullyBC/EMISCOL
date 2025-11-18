# 🛰️ EMISCOL: Scripts de Modelación (CO₂ y CH₄) para Colombia

## 📝 Resumen del Proyecto

Este repositorio contiene los *scripts* de Python para la **modelación inversa y el procesamiento de datos geoespaciales** utilizados en el desarrollo de **EMISCOL**, un sistema de monitoreo dinámico espacial para cuantificar y seguir las emisiones de CO₂ y CH₄ de Colombia.

El objetivo fue superar las limitaciones de los Inventarios Nacionales de GEI (INGEI) en precisión, resolución espacial y desfase temporal.

### 🎯 Metodología Implementada (Contenida en estos Scripts)

El código implementa la lógica del core del análisis:

1.  **Modelación de Gases:** Uso de métodos inversos robustos (**modelo gaussiano** para CO₂ e **IME** para CH₄) utilizando datos satelitales (**OCO-2** y **Sentinel-5P**).
2.  **Procesamiento de Datos:** Manipulación y análisis de datos geoespaciales (NetCDF, raster y vectoriales) para preparar las variables de entrada.

---

## 💻 Dependencias y Stack Tecnológico

Este repositorio solo contiene el *core* de la modelación.

### ⚙️ Herramientas de Modelación (Requeridas para Ejecutar los Scripts)

| Componente | Herramientas Clave | Función Principal |
| :--- | :--- | :--- |
| **Lenguaje** | **Python 3.13.7** | Ejecución de la lógica de modelación. |
| **Análisis de Datos** | **Pandas, NumPy, SciPy, Scikit-learn, StatsModels** | Manipulación de datos, cálculos estadísticos y ML. |
| **Geoespacial** | **Geopandas, rasterio, xarray, shapely** | Procesamiento de datos satelitales (NetCDF, Raster, Vectorial). |

---

## 🚀 Uso e Instalación

1.  **Clonar el Repositorio:**
    ```bash
    git clone [https://docs.github.com/es/repositories/creating-and-managing-repositories/quickstart-for-repositories](https://docs.github.com/es/repositories/creating-and-managing-repositories/quickstart-for-repositories)
    cd EMISCOL-Scripts
    ```
2.  **Configuración del Entorno Python:**
    *Se recomienda crear un entorno virtual e instalar las librerías listadas en el archivo `requirements.txt` (si lo incluyes).*
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚖️ Licencia y Citación

Este código se distribuye bajo la licencia **MIT**.

### 🤝 Cómo Citar este Trabajo

Si utiliza el código, metodología, o resultados derivados de estos scripts en una publicación, solicitamos la **citación formal** del trabajo de tesis/investigación asociado:

> **Balanta, Z. (2025). SISTEMA DE MONITOREO DINÁMICO ESPACIAL PARA LA CUANTIFICACIÓN Y SEGUIMIENTO DE LAS EMISIONES DE CO₂ Y CH₄ DE COLOMBIA. Universidad del Valle, Colombia.**
