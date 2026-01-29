# modules/selectors.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from modules import db_manager

def render_selector_espacial():
    """
    Renderiza un selector espacial UNIFICADO y conectado a Base de Datos.
    Retorna: ids_estaciones, nombre_zona, altitud_ref, gdf_zona_seleccionada
    """
    engine = db_manager.get_engine()
    
    st.sidebar.header("📍 Filtros Geográficos")
    
    # --- 1. MODO DE AGREGACIÓN ---
    modo = st.sidebar.radio(
        "Nivel de Agregación:",
        ["Por Cuenca", "Por Municipio", "Departamento (Antioquia)"],
        index=0
    )
    
    gdf_zona = None
    nombre_zona = "Antioquia"
    altitud_ref = 1500
    
    # --- A. POR CUENCA (CON SELECTOR DE COLUMNA) ---
    if modo == "Por Cuenca":
        try:
            # 1. Consultar columnas disponibles en 'cuencas'
            # Esto es clave: permite ver qué diablos hay realmente en la tabla
            cols_query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'cuencas' AND column_name != 'geometry'"
            df_cols = pd.read_sql(cols_query, engine)
            lista_cols = df_cols['column_name'].tolist()
            
            # Prioridad de selección automática
            default_idx = 0
            for candidata in ['n_nss3', 'subc_lbl', 'nombre_cuenca', 'nombre']:
                if candidata in lista_cols:
                    default_idx = lista_cols.index(candidata)
                    break
            
            # 2. SELECTOR DE CAMPO (La solución a tu duda)
            col_nombre = st.sidebar.selectbox(
                "🗂️ Columna de Nombres:", 
                lista_cols, 
                index=default_idx,
                help="Elige qué columna de la BD usar para listar las cuencas (ej: n_nss3 para tramos)."
            )
            
            # 3. Cargar lista de cuencas usando esa columna
            q_cuencas = f"SELECT {col_nombre}, geometry FROM cuencas ORDER BY {col_nombre}"
            gdf_cuencas = gpd.read_postgis(q_cuencas, engine, geom_col="geometry")
            
            # Limpieza básica
            gdf_cuencas = gdf_cuencas.dropna(subset=[col_nombre])
            lista_nombres = gdf_cuencas[col_nombre].astype(str).unique().tolist()
            lista_nombres.sort()
            
            # 4. Selector de Cuenca
            seleccion = st.sidebar.selectbox("Seleccione Cuenca:", lista_nombres)
            
            if seleccion:
                nombre_zona = seleccion
                gdf_zona = gdf_cuencas[gdf_cuencas[col_nombre].astype(str) == seleccion].head(1)
                
                # Intentar calcular altitud media del polígono seleccionado (si es posible rápido)
                # Si no, dejamos 1500 por defecto
                pass
                
        except Exception as e:
            st.sidebar.error(f"Error cargando cuencas: {e}")

    # --- B. POR MUNICIPIO ---
    elif modo == "Por Municipio":
        try:
            # Lógica simplificada para municipios
            q_mun = "SELECT nombre_municipio, geometry FROM municipios ORDER BY nombre_municipio"
            gdf_mun = gpd.read_postgis(q_mun, engine, geom_col="geometry")
            
            # Fallback si nombre_municipio no existe
            col_nom_mun = 'nombre_municipio' if 'nombre_municipio' in gdf_mun.columns else gdf_mun.columns[0]
            
            lista_mun = gdf_mun[col_nom_mun].unique().tolist()
            seleccion_mun = st.sidebar.selectbox("Seleccione Municipio:", lista_mun)
            
            if seleccion_mun:
                nombre_zona = seleccion_mun
                gdf_zona = gdf_mun[gdf_mun[col_nom_mun] == seleccion_mun].head(1)
                
        except Exception as e:
            st.sidebar.error(f"Error cargando municipios: {e}")

    # --- C. DEPARTAMENTO ---
    else:
        st.sidebar.info("Análisis Regional Completo")
        # gdf_zona sigue siendo None o cargamos el contorno de Antioquia si existe
        
    # --- BUFFER GLOBAL (Opcional pero útil) ---
    buffer_km = st.sidebar.slider("Radio Buffer (km):", 0, 50, 0, help="Expandir zona de búsqueda de estaciones")
    
    # Retorno seguro
    ids_estaciones = [] # El cálculo de estaciones se hace fuera o aquí si quisieras moverlo
    
    # Procesar Buffer si existe zona
    if gdf_zona is not None and buffer_km > 0:
        if gdf_zona.crs.to_string() != "EPSG:3116":
            gdf_zona_m = gdf_zona.to_crs("EPSG:3116")
            gdf_buffer = gdf_zona_m.buffer(buffer_km * 1000)
            gdf_zona = gdf_buffer.to_crs("EPSG:4326")
        else:
            gdf_zona = gdf_zona.buffer(buffer_km * 1000)
            
    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona