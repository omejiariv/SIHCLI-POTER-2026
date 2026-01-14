import streamlit as st
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text

def get_db_engine():
    return create_engine(st.secrets["DATABASE_URL"])

def render_selector_espacial():
    st.sidebar.header("📍 Filtros Geográficos")
    
    # 1. Selector de Nivel
    nivel = st.sidebar.radio(
        "Nivel de Agregación:",
        ["Por Cuenca", "Por Municipio", "Por Región", "Departamento (Antioquia)"]
    )
    
    engine = get_db_engine()
    ids_seleccionados = []
    nombre_seleccion = ""
    altitud_ref = 1500 # Default
    gdf_zona = None
    
    try:
        if nivel == "Por Cuenca":
            # Cargar lista de cuencas
            df_opts = pd.read_sql("SELECT id_cuenca, nombre FROM cuencas ORDER BY nombre", engine)
            sel = st.sidebar.selectbox("Seleccione Cuenca:", df_opts['nombre'])
            if sel:
                nombre_seleccion = f"Cuenca {sel}"
                # Obtener geometría
                q = text("SELECT * FROM cuencas WHERE nombre = :n")
                gdf_zona = gpd.read_postgis(q, engine, params={"n": sel}, geom_col="geometry")
                ids_seleccionados = [int(df_opts[df_opts['nombre']==sel].iloc[0]['id_cuenca'])]

        elif nivel == "Por Municipio":
            # Cargar lista de municipios
            df_opts = pd.read_sql("SELECT id_municipio, nombre FROM municipios ORDER BY nombre", engine)
            sel = st.sidebar.selectbox("Seleccione Municipio:", df_opts['nombre'])
            if sel:
                nombre_seleccion = f"Mpio. {sel}"
                q = text("SELECT * FROM municipios WHERE nombre = :n")
                gdf_zona = gpd.read_postgis(q, engine, params={"n": sel}, geom_col="geometry")
                # Buscamos estaciones dentro del municipio si es necesario
                ids_seleccionados = [] # Aquí podrías hacer una query espacial para buscar estaciones dentro

        elif nivel == "Por Región":
            # Cargar regiones (ej: Oriente, Norte, Valle de Aburrá)
            df_opts = pd.read_sql("SELECT id_region, nombre FROM regiones ORDER BY nombre", engine)
            sel = st.sidebar.selectbox("Seleccione Región:", df_opts['nombre'])
            if sel:
                nombre_seleccion = f"Región {sel}"
                q = text("SELECT * FROM regiones WHERE nombre = :n")
                gdf_zona = gpd.read_postgis(q, engine, params={"n": sel}, geom_col="geometry")

        elif nivel == "Departamento (Antioquia)":
            st.sidebar.info("⚠️ Cargar todo el departamento puede ser lento.")
            nombre_seleccion = "Antioquia"
            # Cargar geometría de Antioquia (asumiendo que está en tabla 'departamentos' o es la unión de municipios)
            # Opción A: Tabla directa
            try:
                gdf_zona = gpd.read_postgis("SELECT * FROM departamentos WHERE nombre='Antioquia'", engine, geom_col="geometry")
            except:
                # Opción B: Fallback (Bounding box general aprox de Antioquia)
                st.warning("Usando BBox genérico de Antioquia.")
                from shapely.geometry import box
                gdf_zona = gpd.GeoDataFrame({'geometry': [box(-77.1, 5.4, -73.9, 8.9)]}, crs="EPSG:4326")

        # Configuración Espacial Común (Buffer)
        if gdf_zona is not None and not gdf_zona.empty:
            # Asegurar CRS WGS84 para visualización
            if gdf_zona.crs != "EPSG:4326":
                gdf_zona = gdf_zona.to_crs("EPSG:4326")
                
            buffer_km = st.sidebar.slider("Radio Buffer (km):", 0, 50, 0, help="Ampliar zona de búsqueda")
            if buffer_km > 0:
                # Proyectar a metros para buffer preciso, luego volver a grados
                gdf_metros = gdf_zona.to_crs("EPSG:3116") 
                gdf_metros['geometry'] = gdf_metros.buffer(buffer_km * 1000)
                gdf_zona = gdf_metros.to_crs("EPSG:4326")
                st.sidebar.success(f"Zona ampliada +{buffer_km}km")

    except Exception as e:
        st.sidebar.error(f"Error de conexión DB: {e}")
    
    return ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona