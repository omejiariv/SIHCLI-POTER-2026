# DOCUMENTACION_TECNICA.md

# 📘 SIHCLI-POTER v3.0: Documentación Técnica y Arquitectura
**Fecha de Actualización:** Enero 2026 | **Versión:** 3.0 Stable

## 1. Visión General del Proyecto
El **Sistema de Información Hidroclimática Integrada (SIHCLI-POTER)** es una plataforma tecnológica diseñada para la gestión integral del recurso hídrico y la biodiversidad en la región Andina. Su propósito es transformar datos dispersos en inteligencia accionable para la toma de decisiones en tiempo real y la planificación a largo plazo.

### Objetivos Clave:
* **Centralización:** Unificar series históricas de precipitación, caudales y datos de biodiversidad en una base de datos relacional segura en la nube.
* **Analítica Avanzada:** Proveer herramientas de cálculo automático (índices climáticos, balances hídricos) y visualización geoespacial.
* **Gestión Operativa:** Facilitar la administración de estaciones, predios y cuencas mediante interfaces CRUD amigables.

---

## 2. Arquitectura del Sistema
SIHCLI-POTER opera bajo una arquitectura híbrida **Cloud-Native**:

### ☁️ Backend (Nube)
* **Base de Datos:** Supabase (PostgreSQL 15) potenciada con la extensión **PostGIS** para el manejo nativo de datos espaciales (geometrías de cuencas y predios).
* **Almacenamiento:** Gestión de grandes volúmenes de datos históricos (millones de registros de precipitación) con políticas de integridad y restricciones únicas.

### 🖥️ Frontend (Cliente)
* **Interfaz de Usuario:** Construida en **Streamlit** (Python), optimizada para interactividad y visualización de datos.
* **Despliegue:** Capacidad de ejecución local u hospedaje en Streamlit Cloud / Docker Containers.

---

## 3. Stack Tecnológico
El ecosistema se basa en Python 3.10+ y utiliza las siguientes librerías core:

| Categoría | Tecnologías / Librerías | Uso Principal |
| :--- | :--- | :--- |
| **Data Science** | `pandas`, `numpy` | Manipulación y limpieza de datos, cálculos estadísticos. |
| **Geospatial** | `geopandas`, `shapely`, `pyproj` | Análisis espacial, reproyección de coordenadas, manejo de SHP/GeoJSON. |
| **Visualización** | `plotly.express`, `folium`, `altair` | Gráficos interactivos, mapas de calor y cartografía dinámica. |
| **Base de Datos** | `sqlalchemy`, `psycopg2-binary` | Conexión ORM y ejecución de consultas SQL optimizadas. |
| **Web Framework** | `streamlit`, `streamlit-aggrid` | Construcción de la interfaz web y tablas interactivas. |

---

## 4. Estructura de Directorios (Mapa de Navegación)
```text
SIHCLI_POTER/
├── .streamlit/          # Configuración del servidor y SECRETOS (credenciales DB)
├── data/                # Archivos estáticos de referencia (GeoJSONs, logos)
├── modules/             # Lógica de Negocio (Backend local)
│   ├── admin_utils.py   # Motor ETL: Carga masiva, limpieza y validación de CSVs
│   ├── data_processor.py# Consultas SQL complejas y funciones de análisis
│   └── utils.py         # Utilidades compartidas (formatos, descargas)
├── pages/               # Módulos de la Aplicación (Pantallas)
│   ├── 01_☁️_Clima...   # Visualización Hidroclimática
│   ├── 09_👑_Panel...   # Panel de Administración (Login protegido)
│   └── ...
├── app.py               # Punto de entrada (Home & Dashboard General)
└── requirements.txt     # Lista de dependencias para instalación