# Módulo de Biodiversidad

import streamlit as st
import sys
import os

# 1. CONFIGURATION
st.set_page_config(page_title="Monitor de Biodiversidad", page_icon="🍃", layout="wide")

try:
    import pandas as pd
    import geopandas as gpd
    import plotly.graph_objects as go
    from sqlalchemy import create_engine, text
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors, config, gbif_connector
except Exception as e:
    st.error(f"Critical import error: {e}")
    st.stop()

st.title("🍃 Biodiversidad y Servicios Ecosistémicos")

# 2. SELECTOR
try:
    ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()
except Exception as e:
    st.error(f"Selector error: {e}")
    st.stop()

# Helper Functions
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
    Loads HYBRID layers: 
    - Local Files: Cuencas, Municipios
    - SQL: Predios
    """
    engine = get_db_engine()
    
    # --- A. FILE-BASED LAYERS (GeoJSON) ---
    file_map = {
        "Cuencas": "SubcuencasAinfluencia.geojson",
        "Municipios": "MunicipiosAntioquia.geojson"
    }
    
    if layer_name in file_map:
        try:
            filename = file_map[layer_name]
            file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', filename))
            
            if os.path.exists(file_path):
                gdf = gpd.read_file(file_path)
                if gdf.crs and gdf.crs != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")
                return gdf
            else:
                st.toast(f"⚠️ File not found: {filename}", icon="📂")
                return None
        except Exception as e:
            print(f"Error loading file {layer_name}: {e}")
            return None

    # --- B. SQL-BASED LAYERS (Predios) ---
    if layer_name == "Predios":
        try:
            table = "predios"
            if gdf_clip_zone is not None:
                minx, miny, maxx, maxy = gdf_clip_zone.total_bounds
                q = text(f"SELECT * FROM {table} WHERE geometry && ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326) LIMIT 2000")
                params = {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}
                return gpd.read_postgis(q, engine, params=params, geom_col="geometry")
            else:
                return gpd.read_postgis(f"SELECT * FROM {table} LIMIT 100", engine, geom_col="geometry")
        except Exception as e:
            print(f"⚠️ SQL Warning ({layer_name}): {e}")
            return None
            
    return None

# 3. MAIN LOGIC
if gdf_zona is not None:
    st.divider()
    
    # --- A. GBIF DOWNLOAD ---
    with st.spinner(f"📡 Scanning biodiversity in {nombre_seleccion}..."):
        gdf_bio = gbif_connector.get_biodiversity_in_polygon(gdf_zona, limit=5000)

    if not gdf_bio.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GBIF Records", f"{len(gdf_bio):,.0f}")
        c2.metric("Species", f"{gdf_bio['Nombre Científico'].nunique():,.0f}")
        
        if c4.button("💾 Save to Local DB"):
            with st.spinner("Writing..."):
                success, msg = save_to_db(gdf_bio)
                if success: st.toast(f"✅ Saved: {msg} records.", icon="💾")
                else: st.error(f"Error: {msg}")

        # --- TABS ---
        tab1, tab2 = st.tabs(["🗺️ Multilayer Map & Parcels", "📊 Taxonomic Analysis"])
        
        with tab1:
            st.markdown("##### Integrated Territorial Viewer")
            col_layers, col_map = st.columns([1, 4])
            
            with col_layers:
                st.subheader("Layers")
                show_munis = st.checkbox("🏛️ Municipalities (GeoJSON)", value=True)
                show_cuencas = st.checkbox("💧 Basins (GeoJSON)", value=True)
                show_predios = st.checkbox("🏡 Parcels (Cadastre)", value=False)
                st.caption("Green points: Biodiversity")

            with col_map:
                fig = go.Figure()

                # 1. Base: Selected Zone
                if gdf_zona is not None:
                    for idx, row in gdf_zona.iterrows():
                        geom = row.geometry
                        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                        for poly in polys:
                            x, y = poly.exterior.xy
                            fig.add_trace(go.Scattermapbox(
                                lon=list(x), lat=list(y), mode='lines', 
                                line=dict(width=2, color='red'), name='Selection Zone', hoverinfo='skip'
                            ))

                # 2. Context (Municipalities - GeoJSON)
                if show_munis:
                    gdf_muni = load_context_layer("Municipios")
                    if gdf_muni is not None and not gdf_muni.empty:
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

                # 3. Context (Basins - GeoJSON)
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
                                # FIXED: Removed dash='dot' (not supported by Scattermapbox)
                                fig.add_trace(go.Scattermapbox(
                                    lon=list(x), lat=list(y), mode='lines',
                                    line=dict(width=1.5, color='blue'), 
                                    name=str(row.get(name_col, 'Subcuenca')), hoverinfo='text'
                                ))
                    elif show_cuencas:
                        st.toast("⚠️ Basins file not found.", icon="📂")

                # 4. Context (Parcels - SQL)
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
                                    name='Parcel', hoverinfo='none', showlegend=False
                                ))
                    else:
                        st.warning("No parcels found.")

                # 5. Biodiversity
                fig.add_trace(go.Scattermapbox(
                    lon=gdf_bio['lon'], lat=gdf_bio['lat'],
                    mode='markers',
                    marker=dict(size=7, color='rgb(0, 200, 100)', opacity=0.8),
                    text=gdf_bio['Nombre Común'],
                    name='Biodiversity (GBIF)'
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
        st.warning("⚠️ No records found in GBIF.")
else:
    st.info("👈 Select a zone.")