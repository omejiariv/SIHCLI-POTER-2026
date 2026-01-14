# Módulo de Biodiversidad

import streamlit as st
import sys
import os

# 1. CONFIGURACIÓN (Debe ser lo primero)
st.set_page_config(page_title="Monitor de Biodiversidad", page_icon="🍃", layout="wide")

try:
    import pandas as pd
    import geopandas as gpd
    import plotly.graph_objects as go
    import plotly.express as px # Necesario para el Sunburst dinámico
    
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

def load_context_layer(layer_name):
    """Carga capas desde archivos GeoJSON locales."""
    # Mapeo de nombres a archivos
    file_map = {
        "Cuencas": "SubcuencasAinfluencia.geojson",
        "Municipios": "MunicipiosAntioquia.geojson",
        "Predios": "PrediosEjecutados.geojson" # CORREGIDO
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
        except Exception as e:
            print(f"Error cargando {layer_name}: {e}")
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
        c3.metric("Familias", f"{gdf_bio['Familia'].nunique():,.0f}" if 'Familia' in gdf_bio.columns else "0")
        
        # Botón descarga
        csv = gdf_bio.drop(columns='geometry', errors='ignore').to_csv(index=False).encode('utf-8')
        c4.download_button("💾 Descargar CSV", csv, "biodiversidad.csv", "text/csv")

        # --- PESTAÑAS ---
        tab1, tab2 = st.tabs(["🗺️ Mapa Multicapa", "📊 Análisis Taxonómico"])
        
        # === TAB 1: MAPA ===
        with tab1:
            st.markdown("##### Visor Territorial Integrado")
            col_layers, col_map = st.columns([1, 4])
            
            with col_layers:
                st.subheader("Capas")
                show_munis = st.checkbox("🏛️ Municipios", value=True)
                show_cuencas = st.checkbox("💧 Cuencas", value=True)
                show_predios = st.checkbox("🏡 Predios", value=False)
                st.caption("Puntos verdes: Biodiversidad")

            with col_map:
                fig = go.Figure()

                # 1. Base: Zona Seleccionada (ROJO)
                if gdf_zona is not None:
                    # Corrección Centroide (Reproyectamos temporalmente para calcular centro)
                    try:
                        centroid = gdf_zona.to_crs('+proj=cea').centroid.to_crs(gdf_zona.crs).iloc[0]
                        center_lat, center_lon = centroid.y, centroid.x
                    except:
                        center_lat, center_lon = 6.2, -75.5 # Fallback Medellin

                    for idx, row in gdf_zona.iterrows():
                        geom = row.geometry
                        # Manejo seguro de MultiPolygon
                        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                        for poly in polys:
                            x, y = poly.exterior.xy
                            fig.add_trace(go.Scattermapbox(
                                lon=list(x), lat=list(y), mode='lines', 
                                line=dict(width=2, color='red'), name='Zona Selección', hoverinfo='skip'
                            ))

                # 2. Contexto (Archivos Locales)
                if show_munis:
                    gdf_muni = load_context_layer("Municipios")
                    if gdf_muni is not None:
                        for _, row in gdf_muni.iterrows():
                            geom = row.geometry
                            if geom:
                                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                                for poly in polys:
                                    x, y = poly.exterior.xy
                                    fig.add_trace(go.Scattermapbox(
                                        lon=list(x), lat=list(y), mode='lines',
                                        line=dict(width=1, color='gray'), hoverinfo='skip'
                                    ))

                if show_cuencas:
                    gdf_cuenca = load_context_layer("Cuencas")
                    if gdf_cuenca is not None:
                         for _, row in gdf_cuenca.iterrows():
                            geom = row.geometry
                            if geom:
                                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                                for poly in polys:
                                    x, y = poly.exterior.xy
                                    # CORREGIDO: Eliminada propiedad 'dash' que causaba error
                                    fig.add_trace(go.Scattermapbox(
                                        lon=list(x), lat=list(y), mode='lines',
                                        line=dict(width=1.5, color='blue'), name='Cuenca', hoverinfo='skip'
                                    ))

                if show_predios:
                    gdf_predios = load_context_layer("Predios")
                    if gdf_predios is not None:
                        # Filtro espacial simple para no saturar
                        gdf_predios = gpd.clip(gdf_predios, gdf_zona) if not gdf_predios.empty else gdf_predios
                        
                        for _, row in gdf_predios.iterrows():
                            geom = row.geometry
                            if geom:
                                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                                for poly in polys:
                                    x, y = poly.exterior.xy
                                    fig.add_trace(go.Scattermapbox(
                                        lon=list(x), lat=list(y), mode='lines',
                                        fill='toself', fillcolor='rgba(255, 165, 0, 0.3)',
                                        line=dict(width=1, color='orange'), 
                                        name='Predio', hoverinfo='text', text="Predio Ejecutado"
                                    ))

                # 3. Biodiversidad (Puntos)
                fig.add_trace(go.Scattermapbox(
                    lon=gdf_bio['lon'], lat=gdf_bio['lat'],
                    mode='markers',
                    marker=dict(size=6, color='rgb(0, 200, 100)'),
                    text=gdf_bio['Nombre Común'],
                    name='Biodiversidad'
                ))

                fig.update_layout(
                    mapbox_style="carto-positron",
                    mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=10),
                    margin={"r":0,"t":0,"l":0,"b":0},
                    height=600,
                    showlegend=False 
                )
                st.plotly_chart(fig, use_container_width=True)

        # === TAB 2: GRÁFICO DINÁMICO (Restaurado) ===
        with tab2:
            st.markdown("##### Estructura Taxonómica")
            if 'Reino' in gdf_bio.columns and 'Familia' in gdf_bio.columns:
                # Limpieza de nulos
                df_chart = gdf_bio.fillna("Sin Dato")
                # Sunburst Interactivo
                fig_sun = px.sunburst(
                    df_chart, 
                    path=['Reino', 'Clase', 'Orden', 'Familia'], 
                    title="Distribución por Grupos Biológicos (Clic para navegar)",
                    height=700
                )
                st.plotly_chart(fig_sun, use_container_width=True)
            else:
                st.warning("Datos insuficientes para el gráfico taxonómico.")
            
            st.markdown("##### Datos Detallados")
            st.dataframe(gdf_bio.drop(columns='geometry', errors='ignore'))

    else:
        st.warning("⚠️ No se encontraron registros en GBIF para esta zona.")
else:
    st.info("👈 Seleccione una zona para comenzar.")