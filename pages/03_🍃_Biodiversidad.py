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
    import plotly.express as px
    
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
def save_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data(ttl=3600)
def load_layer_cached(layer_name):
    """Carga capas GeoJSON locales usando caché para velocidad."""
    file_map = {
        "Cuencas": "SubcuencasAinfluencia.geojson",
        "Municipios": "MunicipiosAntioquia.geojson",
        "Predios": "PrediosEjecutados.geojson"
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
        except Exception:
            return None
    return None

# 3. LÓGICA PRINCIPAL
if gdf_zona is not None:
    st.divider()
    
    # --- A. DESCARGA GBIF ---
    with st.spinner(f"📡 Escaneando biodiversidad en {nombre_seleccion}..."):
        gdf_bio = gbif_connector.get_biodiversity_in_polygon(gdf_zona, limit=3000)

    if not gdf_bio.empty:
        # Métricas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros GBIF", f"{len(gdf_bio):,.0f}")
        c2.metric("Especies", f"{gdf_bio['Nombre Científico'].nunique():,.0f}")
        c3.metric("Familias", f"{gdf_bio['Familia'].nunique():,.0f}" if 'Familia' in gdf_bio.columns else "0")
        
        n_threat = 0
        threatened = pd.DataFrame()
        if 'Amenaza IUCN' in gdf_bio.columns:
            threatened = gdf_bio[~gdf_bio['Amenaza IUCN'].isin(['NE', 'LC', 'NT', 'DD', 'nan'])]
            n_threat = threatened['Nombre Científico'].nunique()
        c4.metric("Amenazadas (IUCN)", f"{n_threat}")

        st.download_button("💾 Descargar Datos (CSV)", save_to_csv(gdf_bio.drop(columns='geometry', errors='ignore')), f"biodiv_{nombre_seleccion}.csv", "text/csv")

        # --- PESTAÑAS ---
        tab1, tab2, tab3 = st.tabs(["🗺️ Mapa Interactivo", "📊 Taxonomía", "🚨 Amenazas"])
        
        # === TAB 1: MAPA OPTIMIZADO ===
        with tab1:
            st.markdown("##### Visor Territorial")
            st.caption("ℹ️ Haz clic en los nombres de la leyenda (derecha) para encender/apagar capas instantáneamente.")
            
            fig = go.Figure()

            # --- 1. CAPA BASE: ZONA SELECCIONADA ---
            if gdf_zona is not None:
                # Centroide
                try:
                    center = gdf_zona.to_crs("+proj=cea").centroid.to_crs("EPSG:4326").iloc[0]
                    center_lat, center_lon = center.y, center.x
                except:
                    center_lat, center_lon = 6.5, -75.5

                for idx, row in gdf_zona.iterrows():
                    geom = row.geometry
                    polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                    for poly in polys:
                        x, y = poly.exterior.xy
                        fig.add_trace(go.Scattermapbox(
                            lon=list(x), lat=list(y), mode='lines', 
                            line=dict(width=3, color='red'), 
                            name='Zona Selección', hoverinfo='skip'
                        ))

            # --- 2. CAPA: MUNICIPIOS ---
            gdf_muni = load_layer_cached("Municipios")
            if gdf_muni is not None:
                # Para evitar agregar miles de trazos, fusionamos visualmente si es posible, 
                # o iteramos con cuidado. Aquí iteramos pero con 'legendgroup' para que se apaguen todos juntos.
                
                # TRUCO DE RENDIMIENTO: Solo dibujamos municipios que interceptan con la vista general
                # (Opcional: Si sigue lento, podemos hacer un clip espacial aquí también)
                
                for idx, row in gdf_muni.iterrows():
                    geom = row.geometry
                    if geom:
                        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                        for i, poly in enumerate(polys):
                            x, y = poly.exterior.xy
                            # Solo añadimos 'name' y 'showlegend' al primer polígono para no spammear la leyenda
                            show_leg = True if idx == 0 and i == 0 else False
                            fig.add_trace(go.Scattermapbox(
                                lon=list(x), lat=list(y), mode='lines',
                                line=dict(width=1, color='gray'), 
                                name='Municipios', 
                                legendgroup='group_muni',
                                showlegend=show_leg,
                                hoverinfo='skip',
                                visible=True # Visible por defecto
                            ))

            # --- 3. CAPA: CUENCAS ---
            gdf_cuenca = load_layer_cached("Cuencas")
            if gdf_cuenca is not None:
                 for idx, row in gdf_cuenca.iterrows():
                    geom = row.geometry
                    if geom:
                        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                        for i, poly in enumerate(polys):
                            x, y = poly.exterior.xy
                            show_leg = True if idx == 0 and i == 0 else False
                            fig.add_trace(go.Scattermapbox(
                                lon=list(x), lat=list(y), mode='lines',
                                line=dict(width=1.5, color='blue'), 
                                name='Cuencas',
                                legendgroup='group_cuenca',
                                showlegend=show_leg,
                                hoverinfo='skip',
                                visible=True # Visible por defecto
                            ))

            # --- 4. CAPA: PREDIOS (Apagada por defecto) ---
            gdf_predios = load_layer_cached("Predios")
            if gdf_predios is not None:
                # Clip espacial ESTRICTO para no cargar predios lejanos
                try:
                    gdf_predios_clip = gpd.clip(gdf_predios, gdf_zona.buffer(0.01))
                except:
                    gdf_predios_clip = gdf_predios

                if not gdf_predios_clip.empty:
                    for idx, row in gdf_predios_clip.iterrows():
                        geom = row.geometry
                        if geom:
                            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                            for i, poly in enumerate(polys):
                                x, y = poly.exterior.xy
                                show_leg = True if idx == 0 and i == 0 else False
                                fig.add_trace(go.Scattermapbox(
                                    lon=list(x), lat=list(y), mode='lines',
                                    fill='toself', fillcolor='rgba(255, 165, 0, 0.4)',
                                    line=dict(width=1, color='orange'), 
                                    name='Predios', 
                                    legendgroup='group_predios',
                                    showlegend=show_leg,
                                    hoverinfo='text', text="Predio",
                                    visible='legendonly' # <--- TRUCO: Apagado al inicio, clic para ver
                                ))

            # --- 5. BIODIVERSIDAD (Encima de todo) ---
            fig.add_trace(go.Scattermapbox(
                lon=gdf_bio['lon'], lat=gdf_bio['lat'],
                mode='markers',
                marker=dict(size=7, color='rgb(0, 200, 100)'),
                text=gdf_bio['Nombre Común'],
                name='Biodiversidad',
                visible=True
            ))

            fig.update_layout(
                mapbox_style="carto-positron",
                mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=10),
                margin={"r":0,"t":0,"l":0,"b":0},
                height=700,
                legend=dict(
                    yanchor="top", y=0.99,
                    xanchor="left", x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        # === TAB 2: TAXONOMÍA ===
        with tab2:
            st.markdown("##### Estructura Taxonómica")
            if 'Reino' in gdf_bio.columns and 'Familia' in gdf_bio.columns:
                df_chart = gdf_bio.fillna("Sin Dato")
                fig_sun = px.sunburst(
                    df_chart, 
                    path=['Reino', 'Clase', 'Orden', 'Familia'], 
                    height=600
                )
                st.plotly_chart(fig_sun, use_container_width=True)
            else:
                st.warning("Datos insuficientes.")
            st.dataframe(gdf_bio.drop(columns='geometry', errors='ignore'))

        # === TAB 3: AMENAZAS ===
        with tab3:
            st.markdown("##### Especies Amenazadas (IUCN)")
            if not threatened.empty:
                st.warning(f"⚠️ {n_threat} especies en riesgo.")
                df_show = threatened[['Nombre Científico', 'Nombre Común', 'Amenaza IUCN', 'Familia']].drop_duplicates()
                st.dataframe(df_show, use_container_width=True)
                
                fig_heat = px.density_mapbox(
                    threatened, lat='lat', lon='lon', radius=25,
                    zoom=10, height=500, title="Mapa de Calor: Amenazas"
                )
                fig_heat.update_layout(mapbox_style="carto-positron")
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.success("✅ No se detectaron especies críticas.")

    else:
        st.warning("⚠️ No se encontraron registros en GBIF para esta zona.")
else:
    st.info("👈 Seleccione una zona para comenzar.")