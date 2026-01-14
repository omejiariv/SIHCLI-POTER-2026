# Módulo de Biodiversidad

import streamlit as st
import sys
import os

# 1. CONFIGURACIÓN
st.set_page_config(page_title="Monitor de Biodiversidad", page_icon="🍃", layout="wide")

try:
    import pandas as pd
    import geopandas as gpd
    import plotly.express as px
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors, config, gbif_connector
except Exception as e:
    st.error(f"🚨 Error crítico de importación: {e}")
    st.stop()

st.title("🍃 Biodiversidad: Datos Globales (GBIF)")

# 2. SELECTOR
try:
    ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()
except Exception as e:
    st.error(f"Error selector: {e}")
    st.stop()

# 3. LÓGICA PRINCIPAL
if gdf_zona is not None:
    st.divider()
    
    # --- SECCIÓN DE DIAGNÓSTICO (Temporal) ---
    with st.expander("🕵️‍♂️ Datos de Depuración (Si ves esto vacío, abre aquí)", expanded=True):
        # Verificar coordenadas
        gdf_wgs84 = gdf_zona.to_crs("EPSG:4326")
        minx, miny, maxx, maxy = gdf_wgs84.total_bounds
        
        st.write(f"**Zona Seleccionada:** {nombre_seleccion}")
        st.write(f"**Coordenadas de Búsqueda (WGS84):**")
        st.code(f"Min X (Lon): {minx}\nMin Y (Lat): {miny}\nMax X (Lon): {maxx}\nMax Y (Lat): {maxy}")
        
        if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
            st.error("🚨 ¡ALERTA! Las coordenadas parecen estar en METROS, no en GRADOS. GBIF necesita grados.")
        else:
            st.success("✅ Coordenadas válidas para GPS.")
            
        st.write(f"**URL de Prueba (GBIF API):**")
        url_test = f"https://api.gbif.org/v1/occurrence/search?decimalLatitude={miny},{maxy}&decimalLongitude={minx},{maxx}&limit=5"
        st.markdown(f"[Haz clic aquí para probar la API en el navegador]({url_test})")

    # --- FIN DIAGNÓSTICO ---

    with st.spinner(f"📡 Buscando especies en {nombre_seleccion}..."):
        try:
            gdf_bio = gbif_connector.get_biodiversity_in_polygon(gdf_zona, limit=3000)
        except Exception as e:
            st.error(f"Error conector: {e}")
            gdf_bio = gpd.GeoDataFrame()

    if not gdf_bio.empty:
        # KPI's
        c1, c2, c3, c4 = st.columns(4)
        n_total = len(gdf_bio)
        n_species = gdf_bio['Nombre Científico'].nunique()
        n_families = gdf_bio['Familia'].nunique() if 'Familia' in gdf_bio.columns else 0
        
        c1.metric("Registros", f"{n_total:,.0f}")
        c2.metric("Especies", f"{n_species:,.0f}")
        c3.metric("Familias", f"{n_families}")
        c4.metric("Amenazadas", "Calculando...")
        
        tab1, tab2 = st.tabs(["🗺️ Mapa", "🧬 Taxonomía"])
        
        with tab1:
            fig_map = px.scatter_mapbox(
                gdf_bio, lat="lat", lon="lon", color="Reino" if "Reino" in gdf_bio.columns else None,
                hover_name="Nombre Científico", zoom=10, height=600
            )
            fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)

        with tab2:
            st.dataframe(gdf_bio.drop(columns='geometry'))

    else:
        st.warning("⚠️ La API respondió, pero no trajo registros.")
        st.write("Posibles causas: La zona es muy pequeña, no hay datos públicos, o el recorte espacial (clip) eliminó los puntos cercanos.")

else:
    st.info("👈 Seleccione una zona.")