# Módulo de Biodiversidad

import streamlit as st
import sys
import os

# 1. CONFIGURACIÓN
st.set_page_config(page_title="Monitor de Biodiversidad", page_icon="🍃", layout="wide")

try:
    import pandas as pd
    import geopandas as gpd
    import plotly.graph_objects as go
    from sqlalchemy import create_engine, text
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors, config, gbif_connector
except Exception as e:
    st.error(f"Error crítico de importación: {e}")
    st.stop()

st.title("🍃 Biodiversidad y Servicios Ecosistémicos")

# 2. SELECTOR
try:
    ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()
except Exception as e:
    st.error(f"Error en selector: {e}")
    st.stop()

# Funciones Auxiliares
def get_db_engine():
    return create_engine(st.secrets["DATABASE_URL"])

def save_to_db(df, table_name="biodiversidad_registros"):
    engine = get_db_engine()
    df_save = df.drop(columns=['geometry'], errors='ignore').copy()
    df_save['fecha_descarga'] = pd.Timestamp.now()
    df_save['origen'] = 'GBIF_API'
    try:
        df_save.to_sql(table_name, engine, if_exists='append', index=False)
        return True, len(df_save)
    except Exception as e:
        return False, str(e)

def load_context_layer(layer_name, gdf_clip_zone=None):
    """
    Carga capas HÍBRIDAS: 
    - Archivos Locales: Cuencas, Municipios
    - SQL: Predios
    """
    engine = get_db_engine()
    
    # --- A. CAPAS BASADAS EN ARCHIVOS (GeoJSON) ---
    # Diccionario: Nombre Capa -> Nombre Archivo en carpeta 'data'
    file_map = {
        "Cuencas": "SubcuencasAinfluencia.geojson",
        "Municipios": "MunicipiosAntioquia.geojson"
    }
    
    if layer_name in file_map:
        try:
            filename = file_map[layer_name]
            # Ruta relativa a la carpeta data/
            file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', filename))
            
            if os.path.exists(file_path):
                # Cargar archivo
                gdf = gpd.read_file(file_path)
                
                # Asegurar proyección web (Lat/Lon)
                if gdf.crs and gdf.crs != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")
                return gdf
            else:
                st.toast(f"⚠️ Archivo no encontrado: {filename}", icon="📂")
                return None
        except Exception as e:
            print(f"Error cargando archivo {layer_name}: {e}")
            return None

    # --- B. CAPAS BASADAS EN SQL (Predios) ---
    if layer_name == "Predios":
        try:
            table = "predios"
            if gdf_clip_zone is not None:
                # Filtro espacial para no traer millones de predios
                minx, miny, maxx, maxy = gdf_clip_zone.total_bounds
                q = text(f"SELECT * FROM {table} WHERE geometry && ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326) LIMIT 2000")
                params = {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}
                return gpd.read_postgis(q, engine, params=params, geom_col="geometry")
            else:
                return gpd.read_postgis(f"SELECT * FROM {table} LIMIT 100", engine, geom_col="geometry")
        except Exception as e:
            print(f"⚠️ Advertencia SQL ({layer_name}): {e}")
            return None
            
    return None

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
        
        if c4.button("💾 Guardar en Base de Datos Local"):
            with st.spinner("Escribiendo..."):
                success, msg = save_to_db(gdf_bio)
                if success: st.toast(f"✅ Guardado: {msg} registros.", icon="💾")
                else: st.error(f"Error: {msg}")

        # --- PESTAÑAS ---
        tab1, tab2 = st.tabs(["🗺️ Mapa Multicapa & Predios", "📊 Análisis Taxonómico"])
        
        with tab1:
            st.markdown("##### Visor Territorial Integrado")
            col_layers, col_map = st.columns([1, 4])
            
            with col_layers:
                st.subheader("Capas")
                show_munis = st.checkbox("🏛️ Municipios (GeoJSON)", value=True)
                show_cuencas = st.checkbox("💧 Cuencas (GeoJSON)", value=True)
                show_predios = st.checkbox("🏡 Predios (Catastro)", value=False)
                st.caption("Puntos verdes: Biodiversidad")

            with col_map:
                fig = go.Figure()

                # 1. Base: Zona Seleccionada
                if gdf_zona is not None:
                    for idx, row in gdf_zona.iterrows():
                        geom = row.geometry
                        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                        for poly in polys:
                            x, y = poly.exterior.xy
                            fig.add_trace(go.Scattermapbox(
                                lon=list(x), lat=list(y), mode='lines', 
                                line=dict(width=2, color='red'), name='Zona Selección', hoverinfo='skip'
                            ))

                # 2. Contexto (Municipios - GeoJSON)
                if show_munis:
                    gdf_muni = load_context_layer("Municipios")
                    if gdf_muni is not None and not gdf_muni.empty:
                        # Buscamos nombre columna
                        name_col = next((c for c in ['MPIO_CNMBR', 'nombre', 'NOMBRE', 'municipio'] if c in gdf_muni.columns), 'Municipio')
                        
                        for _, row in gdf_muni.iterrows():
                            geom = row.geometry
                            if geom is None: continue
                            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                            for poly in polys:
                                x, y = poly.exterior.xy
                                fig.add_trace(go.Scattermapbox(
                                    lon=list(x), lat=list(y), mode='lines',
                                    line=dict(width=1, color='gray'), 
                                    name=str(row.get(name_col, 'Mpio')), hoverinfo='text'
                                ))

                # 3. Contexto (Cuencas - GeoJSON)
                if show_cuencas:
                    gdf_cuenca = load_context_layer("Cuencas")
                    if gdf_cuenca is not None and not gdf_cuenca.empty:
                         name_col = next((c for c in ['nombre', 'Name', 'NOM_CUENCA', 'subcuenca'] if c in gdf_cuenca.columns), 'Cuenca')
                         
                         for _, row in gdf_cuenca.iterrows():
                            geom = row.geometry
                            if geom is None: continue
                            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                            
                            for poly in polys:
                                x, y = poly.exterior.xy
                                fig.add_trace(go.Scattermapbox(
                                    lon=list(x), lat=list(y), mode='lines',
                                    line=dict(width=1.5, color='blue', dash='dot'), 
                                    name=str(row.get(name_col, 'Subcuenca')), hoverinfo='text'
                                ))
                    elif show_cuencas:
                        st.toast("⚠️ Archivo de Cuencas no encontrado.", icon="📂")

                # 4. Contexto (Predios - SQL)
                if show_predios:
                    gdf_predios = load_context_layer("Predios", gdf_zona)
                    if gdf_predios is not None and not gdf_predios.empty:
                        for _, row in gdf_predios.iterrows():
                            geom = row.geometry
                            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                            for poly in polys:
                                x, y = poly.exterior.xy
                                fig.add_trace(go.Scattermapbox(
                                    lon=list(x), lat=list(y), mode='lines',
                                    fill='toself', fillcolor='rgba(255, 165, 0, 0.1)',
                                    line=dict(width=0.5, color='orange'), 
                                    name='Predio', hoverinfo='none', showlegend=False
                                ))
                    else:
                        st.warning("No se encontraron predios.")

                # 5. Biodiversidad
                fig.add_trace(go.Scattermapbox(
                    lon=gdf_bio['lon'], lat=gdf_bio['lat'],
                    mode='markers',
                    marker=dict(size=7, color='rgb(0, 200, 100)', opacity=0.8),
                    text=gdf_bio['Nombre Común'],
                    name='Biodiversidad (GBIF)'
                ))

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