# modules/selectors.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from modules import db_manager
from modules.config import Config

def render_selector_espacial():
    """
    Selector espacial UNIFICADO, BLINDADO y con AUTO-CORRECCIÓN de Coordenadas.
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
    
    try:
        # --- A. POR CUENCA ---
        if modo == "Por Cuenca":
            try:
                # Leemos la geometría
                gdf_cuencas = gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry")
                
                # Detectar columna de nombre (priorizamos tus nombres comunes)
                posibles_nombres = ['subc_lbl', 'nombre', 'nombre_cuenca', 'cuenca']
                col_nom = next((c for c in gdf_cuencas.columns if c in posibles_nombres), None)
                
                if col_nom:
                    lista = sorted(gdf_cuencas[col_nom].astype(str).unique().tolist())
                    sel = st.sidebar.selectbox("Seleccione Cuenca:", lista)
                    if sel:
                        nombre_zona = sel
                        gdf_zona = gdf_cuencas[gdf_cuencas[col_nom] == sel]
                else:
                    st.sidebar.error(f"No encontré columna de nombre en cuencas. Columnas: {gdf_cuencas.columns.tolist()}")

            except Exception as e:
                st.sidebar.warning(f"Error cargando cuencas: {e}")
                return [], "", 0, None

        # --- B. POR MUNICIPIO ---
        elif modo == "Por Municipio":
            try:
                gdf_mun = gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry")
                
                posibles_nombres = ['mpio_cnmbr', 'nombre', 'municipio', 'nombre_municipio']
                col_nom = next((c for c in gdf_mun.columns if c in posibles_nombres), None)
                
                if col_nom:
                    lista = sorted(gdf_mun[col_nom].astype(str).unique().tolist())
                    sel = st.sidebar.selectbox("Seleccione Municipio:", lista)
                    if sel:
                        nombre_zona = sel
                        gdf_zona = gdf_mun[gdf_mun[col_nom] == sel]
            except:
                st.sidebar.warning("Tabla 'municipios' no disponible o con error.")

        # --- C. DEPARTAMENTO ---
        else:
            from shapely.geometry import box
            # Caja aproximada de Antioquia en Lat/Lon
            gdf_zona = gpd.GeoDataFrame(
                {'nombre': ['Antioquia']}, 
                geometry=[box(-77.5, 5.0, -73.5, 9.0)], 
                crs="EPSG:4326"
            )

        # --- 2. FILTRAR ESTACIONES (EL CEREBRO DEL MAPA) ---
        ids_estaciones = []
        if gdf_zona is not None and not gdf_zona.empty:
            
            # --- AUTO-CORRECCIÓN DE COORDENADAS (CRÍTICO) ---
            # Si las coordenadas son gigantes (ej: 800,000), es Magna Sirgas. Hay que pasar a Lat/Lon.
            bounds = gdf_zona.total_bounds
            if abs(bounds[0]) > 180: 
                # st.sidebar.info("🔄 Detecté coordenadas planas. Convirtiendo a GPS (Lat/Lon)...")
                try:
                    # Asumimos Magna Sirgas origen Nacional (EPSG:3116) o Bogotá (EPSG:21818)
                    # EPSG:3116 es el estándar moderno para Colombia.
                    gdf_zona = gdf_zona.set_crs("EPSG:3116", allow_override=True)
                    gdf_zona = gdf_zona.to_crs("EPSG:4326")
                except Exception as e:
                    st.sidebar.error(f"Error reproyectando: {e}")
            else:
                # Si ya son pequeñas, aseguramos que sepa que es 4326
                if not gdf_zona.crs:
                    gdf_zona = gdf_zona.set_crs("EPSG:4326")

            # Slider de Buffer (Radio de Búsqueda)
            buff_km = st.sidebar.slider("Radio Buffer (km):", 0, 50, 5)
            # Convertimos km a grados aprox (1 grado ~ 111km)
            buff_deg = buff_km / 111.0 
            
            # Recalcular limites con el buffer
            minx, miny, maxx, maxy = gdf_zona.total_bounds
            
            # CONSULTA SQL GEOGRÁFICA
            # Buscamos estaciones dentro del cuadro delimitador + buffer
            q_est = text(f"""
                SELECT id_estacion, nombre, latitud, longitud, altitud 
                FROM estaciones 
                WHERE longitud BETWEEN {minx - buff_deg} AND {maxx + buff_deg} 
                AND latitud BETWEEN {miny - buff_deg} AND {maxy + buff_deg}
            """)
            
            df_est = pd.read_sql(q_est, engine)
            
            if not df_est.empty:
                # Filtro Fino: Usamos geometría exacta (Punto dentro de Polígono)
                gdf_ptos = gpd.GeoDataFrame(
                    df_est, 
                    geometry=gpd.points_from_xy(df_est.longitud, df_est.latitud), 
                    crs="EPSG:4326"
                )
                
                # Aplicamos buffer a la geometría de la zona para el filtro exacto
                zona_buffered = gdf_zona.to_crs("EPSG:3116").buffer(buff_km * 1000).to_crs("EPSG:4326").unary_union
                
                est_in = gdf_ptos[gdf_ptos.geometry.within(zona_buffered)]
                
                if not est_in.empty:
                    ids_estaciones = est_in['id_estacion'].astype(str).str.strip().tolist()
                    altitud_ref = est_in['altitud'].mean()
                    st.sidebar.success(f"📍 Estaciones encontradas: {len(ids_estaciones)}")
                else:
                    st.sidebar.warning("Estaciones cercanas, pero fuera del polígono exacto.")
            else:
                st.sidebar.warning(f"No hay estaciones en esta zona.")
                # Debug (Opcional): Mostrar qué buscó
                # st.sidebar.code(f"Buscando Lon: {minx:.2f} a {maxx:.2f}")

    except Exception as e:
        st.sidebar.error(f"Error selector: {e}")
        
    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona