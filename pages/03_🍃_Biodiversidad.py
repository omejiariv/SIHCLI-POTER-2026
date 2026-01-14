# Módulo de Biodiversidad

import streamlit as st
import sys
import os

# 1. CONFIGURACIÓN DE PÁGINA (SIEMPRE PRIMERO)
st.set_page_config(page_title="Monitor de Biodiversidad", page_icon="🍃", layout="wide")

# 2. IMPORTS SEGUROS
try:
    import pandas as pd
    import geopandas as gpd
    import plotly.express as px
    
    # Rutas Modulares
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors, config
    from modules import gbif_connector # Conector API
except Exception as e:
    st.error(f"🚨 Error crítico de importación: {e}")
    st.stop()

st.title("🍃 Biodiversidad: Datos Globales (GBIF)")
st.markdown("Inventario de biodiversidad en tiempo real usando la API de GBIF y datos de la cuenca.")

# --- 3. SELECTOR GEOGRÁFICO ---
try:
    ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()
except Exception as e:
    st.error(f"Error cargando selector: {e}")
    st.stop()

# --- 4. LÓGICA PRINCIPAL ---
if gdf_zona is not None:
    st.divider()
    
    # Carga de Datos
    with st.spinner(f"📡 Buscando especies en {nombre_seleccion}..."):
        try:
            gdf_bio = gbif_connector.get_biodiversity_in_polygon(gdf_zona, limit=3000)
        except Exception as e:
            st.error(f"Error conectando con GBIF: {e}")
            gdf_bio = gpd.GeoDataFrame()

    if not gdf_bio.empty:
        # --- KPI's ---
        c1, c2, c3, c4 = st.columns(4)
        n_total = len(gdf_bio)
        n_species = gdf_bio['Nombre Científico'].nunique()
        
        # Familias (Manejo de errores si la columna no llega)
        n_families = gdf_bio['Familia'].nunique() if 'Familia' in gdf_bio.columns else 0
        
        # Amenazas
        if 'Amenaza IUCN' in gdf_bio.columns:
            threatened = gdf_bio[~gdf_bio['Amenaza IUCN'].isin(['NE', 'LC', 'NT', 'DD', 'nan'])]
            n_threat = threatened['Nombre Científico'].nunique()
        else:
            threatened = pd.DataFrame()
            n_threat = 0
        
        c1.metric("Registros", f"{n_total:,.0f}")
        c2.metric("Especies", f"{n_species:,.0f}")
        c3.metric("Familias", f"{n_families}")
        c4.metric("Amenazadas (IUCN)", f"{n_threat}")
        
        # --- PESTAÑAS ---
        tab1, tab2, tab3 = st.tabs(["🗺️ Mapa", "🧬 Taxonomía", "🚨 Amenazas"])
        
        with tab1:
            # Mapa
            color_col = "Reino" if "Reino" in gdf_bio.columns else None
            fig_map = px.scatter_mapbox(
                gdf_bio, lat="lat", lon="lon", 
                color=color_col,
                hover_name="Nombre Común" if "Nombre Común" in gdf_bio.columns else "Nombre Científico",
                zoom=10, height=600
            )
            fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)

        with tab2:
            # Sunburst
            if 'Reino' in gdf_bio.columns and 'Familia' in gdf_bio.columns:
                df_sun = gdf_bio.fillna("Desconocido")
                fig_sun = px.sunburst(
                    df_sun, path=['Reino', 'Clase', 'Orden', 'Familia'], 
                    height=700, title="Jerarquía Taxonómica"
                )
                st.plotly_chart(fig_sun, use_container_width=True)
            else:
                st.warning("Faltan datos taxonómicos para el gráfico solar.")
            
            # Descarga
            st.download_button(
                "💾 Descargar CSV", 
                gdf_bio.drop(columns='geometry').to_csv(index=False).encode('utf-8'), 
                f"bio_{nombre_seleccion}.csv"
            )

        with tab3:
            if not threatened.empty:
                st.dataframe(threatened[['Nombre Científico', 'Nombre Común', 'Amenaza IUCN', 'Familia']].drop_duplicates())
            else:
                st.success("No se detectaron especies en categorías críticas (VU, EN, CR).")

    else:
        st.warning("⚠️ No se encontraron registros en GBIF para esta zona específica.")
else:
    st.info("👈 Seleccione una zona en el menú lateral.")