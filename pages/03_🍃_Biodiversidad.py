# Módulo de Biodiversidad

import streamlit as st
import sys
import os

# 1. CONFIGURACIÓN
st.set_page_config(page_title="Monitor de Biodiversidad", page_icon="🍃", layout="wide")

try:
    import pandas as pd
    import geopandas as gpd
    import plotly.graph_objects as go # Usamos Graph Objects para control total de capas
    from sqlalchemy import create_engine, text
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors, config, gbif_connector
except Exception as e:
    st.error(f"Error crítico de importación: {e}")
    st.stop()

st.title("🍃 Biodiversidad y Servicios Ecosistémicos")

# 2. SELECTOR (Ahora con Municipios/Regiones funcionales)
try:
    ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()
except Exception as e:
    st.error(f"Error en selector: {e}")
    st.stop()

# Funciones Auxiliares DB
def get_db_engine():
    return create_engine(st.secrets["DATABASE_URL"])

def save_to_db(df, table_name="biodiversidad_registros"):
    """Guarda el dataframe en la base de datos local."""
    engine = get_db_engine()
    # Limpieza antes de guardar (eliminar columnas geométricas complejas si las hay)
    df_save = df.drop(columns=['geometry'], errors='ignore').copy()
    
    # Agregar timestamp de descarga
    df_save['fecha_descarga'] = pd.Timestamp.now()
    df_save['origen'] = 'GBIF_API'
    
    try:
        df_save.to_sql(table_name, engine, if_exists='append', index=False)
        return True, len(df_save)
    except Exception as e:
        return False, str(e)

def load_context_layer(layer_name, gdf_clip_zone=None):
    """Carga capas de contexto (Municipios, Predios) recortadas a la zona actual."""
    engine = get_db_engine()
    table_map = {
        "Municipios": "municipios",
        "Cuencas": "cuencas",
        "Predios": "predios" # Cuidado: Puede ser pesado
    }
    
    if layer_name not in table_map: return None
    
    table = table_map[layer_name]
    
    # Si es 'Predios', hacemos una query espacial estricta para no traernos toda la base
    if layer_name == "Predios" and gdf_clip_zone is not None:
        # Obtenemos el BBOX para filtrar en SQL (Más rápido)
        minx, miny, maxx, maxy = gdf_clip_zone.total_bounds
        q = text(f"""
            SELECT * FROM {table} 
            WHERE geometry && ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326)
            LIMIT 2000 -- Límite de seguridad
        """)
        params = {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}
        return gpd.read_postgis(q, engine, params=params, geom_col="geometry")
    
    # Para capas ligeras (Municipios/Cuencas)
    return gpd.read_postgis(f"SELECT * FROM {table}", engine, geom_col="geometry")

# 3. LÓGICA PRINCIPAL
if gdf_zona is not None:
    st.divider()
    
    # --- A. DESCARGA GBIF ---
    with st.spinner(f"📡 Escaneando biodiversidad en {nombre_seleccion}..."):
        gdf_bio = gbif_connector.get_biodiversity_in_polygon(gdf_zona, limit=5000)

    if not gdf_bio.empty:
        # Métricas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros GBIF", f"{len(gdf_bio):,.0f}")
        c2.metric("Especies", f"{gdf_bio['Nombre Científico'].nunique():,.0f}")
        
        # Botón de Guardado (Requerimiento 1)
        if c4.button("💾 Guardar en Base de Datos Local"):
            with st.spinner("Escribiendo en base de datos..."):
                success, msg = save_to_db(gdf_bio)
                if success:
                    st.toast(f"✅ ¡Éxito! {msg} registros guardados.", icon="💾")
                else:
                    st.error(f"Error al guardar: {msg}")

        # --- PESTAÑAS ---
        tab1, tab2 = st.tabs(["🗺️ Mapa Multicapa & Predios", "📊 Análisis Taxonómico"])
        
        with tab1:
            st.markdown("##### Visor Territorial Integrado")
            
            # Selector de Capas (Requerimiento 3)
            col_layers, col_map = st.columns([1, 4])
            
            with col_layers:
                st.subheader("Capas")
                show_munis = st.checkbox("🏛️ Municipios", value=True)
                show_cuencas = st.checkbox("💧 Cuencas", value=True)
                show_predios = st.checkbox("🏡 Predios (Catastro)", value=False, help="Muestra los predios dentro de la zona (Máx 2000).")
                
                st.caption("Los puntos de colores son la biodiversidad.")

            with col_map:
                # Construcción del Mapa con Graph Objects (Más flexible)
                fig = go.Figure()

                # 1. Capa Base: Zona Seleccionada (Borde Rojo)
                if gdf_zona is not None:
                    for geom in gdf_zona.geometry:
                        if geom.geom_type == 'Polygon': x, y = geom.exterior.xy
                        else: x, y = geom.exterior.xy # Simplificado para demo
                        fig.add_trace(go.Scattermapbox(
                            lon=list(x), lat=list(y), mode='lines', 
                            line=dict(width=2, color='red'), name='Zona Selección'
                        ))

                # 2. Capas de Contexto (Municipios/Cuencas/Predios)
                if show_munis:
                    gdf_muni = load_context_layer("Municipios")
                    if gdf_muni is not None and not gdf_muni.empty:
                        # Dibujamos líneas grises
                        for _, row in gdf_muni.iterrows():
                            geom = row.geometry
                            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                                # Simplificación rápida para visualización
                                if geom.geom_type == 'Polygon': polys = [geom]
                                else: polys = geom.geoms
                                for poly in polys:
                                    x, y = poly.exterior.xy
                                    fig.add_trace(go.Scattermapbox(
                                        lon=list(x), lat=list(y), mode='lines',
                                        line=dict(width=1, color='gray'), 
                                        name=f"Mpio: {row.get('nombre', 'Mpio')}",
                                        hoverinfo='text'
                                    ))

                if show_cuencas:
                    gdf_cuenca = load_context_layer("Cuencas")
                    if gdf_cuenca is not None and not gdf_cuenca.empty:
                         for _, row in gdf_cuenca.iterrows():
                            geom = row.geometry
                            if geom.geom_type == 'Polygon': polys = [geom]
                            else: polys = geom.geoms
                            for poly in polys:
                                x, y = poly.exterior.xy
                                fig.add_trace(go.Scattermapbox(
                                    lon=list(x), lat=list(y), mode='lines',
                                    line=dict(width=1, color='blue', dash='dot'), 
                                    name=f"Cuenca: {row.get('nombre', 'Cuenca')}",
                                    hoverinfo='text'
                                ))

                if show_predios:
                    gdf_predios = load_context_layer("Predios", gdf_zona)
                    if gdf_predios is not None and not gdf_predios.empty:
                        for _, row in gdf_predios.iterrows():
                            geom = row.geometry
                            if geom.geom_type == 'Polygon': polys = [geom]
                            else: polys = geom.geoms
                            for poly in polys:
                                x, y = poly.exterior.xy
                                fig.add_trace(go.Scattermapbox(
                                    lon=list(x), lat=list(y), mode='lines',
                                    fill='toself', fillcolor='rgba(255, 165, 0, 0.1)', # Naranja muy transparente
                                    line=dict(width=0.5, color='orange'), 
                                    name='Predio', hoverinfo='none', showlegend=False
                                ))
                    else:
                        st.warning("No se encontraron predios en esta zona o capa vacía.")

                # 3. Capa Biodiversidad (Puntos)
                fig.add_trace(go.Scattermapbox(
                    lon=gdf_bio['lon'], lat=gdf_bio['lat'],
                    mode='markers',
                    marker=dict(size=7, color='rgb(0, 200, 100)', opacity=0.8),
                    text=gdf_bio['Nombre Común'],
                    name='Biodiversidad (GBIF)'
                ))

                # Ajustes Finales Mapa
                center_lat = gdf_zona.geometry.centroid.y.mean()
                center_lon = gdf_zona.geometry.centroid.x.mean()
                
                fig.update_layout(
                    mapbox_style="carto-positron",
                    mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=10),
                    margin={"r":0,"t":0,"l":0,"b":0},
                    height=600,
                    legend=dict(orientation="h", y=0, x=0, bgcolor="rgba(255,255,255,0.8)")
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.dataframe(gdf_bio.drop(columns='geometry'))

    else:
        st.warning("⚠️ No se encontraron registros en GBIF.")
else:
    st.info("👈 Seleccione una zona.")