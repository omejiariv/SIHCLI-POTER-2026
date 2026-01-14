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
    st.error(f"Error de sistema: {e}")
    st.stop()

st.title("🍃 Biodiversidad: Datos Globales (GBIF)")
st.markdown("Inventario de biodiversidad en tiempo real. Fuente: Global Biodiversity Information Facility API.")

# 2. SELECTOR
try:
    ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()
except:
    st.stop()

# 3. LÓGICA PRINCIPAL
if gdf_zona is not None:
    st.divider()
    
    # Carga de Datos
    with st.spinner(f"📡 Escaneando biodiversidad en {nombre_seleccion}..."):
        # Aumentamos el límite para intentar capturar más diversidad si existe
        gdf_bio = gbif_connector.get_biodiversity_in_polygon(gdf_zona, limit=5000)

    if not gdf_bio.empty:
        # --- KPI's (INDICADORES) ---
        c1, c2, c3, c4 = st.columns(4)
        
        n_total = len(gdf_bio)
        n_species = gdf_bio['Nombre Científico'].nunique()
        n_families = gdf_bio['Familia'].nunique() if 'Familia' in gdf_bio.columns else 0
        
        # Cálculo de Amenazas
        n_threat = 0
        threatened = pd.DataFrame()
        if 'Amenaza IUCN' in gdf_bio.columns:
            # Filtramos lo que NO es amenaza (LC=Preocupación Menor, NE=No Evaluado, etc.)
            threatened = gdf_bio[~gdf_bio['Amenaza IUCN'].isin(['NE', 'LC', 'NT', 'DD', 'nan'])]
            n_threat = threatened['Nombre Científico'].nunique()
        
        c1.metric("Registros Totales", f"{n_total:,.0f}")
        c2.metric("Riqueza de Especies", f"{n_species:,.0f}")
        c3.metric("Familias Biológicas", f"{n_families}")
        c4.metric("Especies Amenazadas", f"{n_threat}", help="Categorías Vulnerable (VU), En Peligro (EN) o Crítico (CR)")
        
        # --- PESTAÑAS ---
        tab1, tab2, tab3 = st.tabs(["🗺️ Mapa de Distribución", "🧬 Taxonomía Visual", "🚨 Estado de Conservación"])
        
        with tab1:
            st.markdown(f"##### Distribución de {n_species} especies en la zona")
            color_col = "Reino" if "Reino" in gdf_bio.columns else None
            hover_name = "Nombre Común" if "Nombre Común" in gdf_bio.columns else "Nombre Científico"
            
            fig_map = px.scatter_mapbox(
                gdf_bio, lat="lat", lon="lon", 
                color=color_col,
                hover_name=hover_name,
                hover_data={"Nombre Científico": True, "Familia": True, "lat": False, "lon": False},
                zoom=10, height=600,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)

        with tab2:
            st.markdown("##### Estructura del Ecosistema")
            if 'Reino' in gdf_bio.columns and 'Familia' in gdf_bio.columns:
                df_sun = gdf_bio.fillna("Desconocido")
                # Gráfico Solar (Sunburst)
                fig_sun = px.sunburst(
                    df_sun, 
                    path=['Reino', 'Clase', 'Orden', 'Familia'], 
                    height=700,
                    color='Reino'
                )
                st.plotly_chart(fig_sun, use_container_width=True)
            else:
                st.info("No hay suficiente información taxonómica para generar el árbol.")
            
            # Tabla Descargable
            with st.expander("📄 Ver Tabla de Datos Completa"):
                st.dataframe(gdf_bio.drop(columns='geometry'))
                st.download_button(
                    "💾 Descargar Inventario (CSV)", 
                    gdf_bio.drop(columns='geometry').to_csv(index=False).encode('utf-8'), 
                    f"biodiversidad_{nombre_seleccion}.csv"
                )

        with tab3:
            st.markdown("##### Especies en Lista Roja (IUCN)")
            if not threatened.empty:
                st.warning(f"⚠️ Se han detectado {n_threat} especies con categoría de amenaza alta.")
                
                # Resumen simple
                df_show = threatened[['Nombre Científico', 'Nombre Común', 'Amenaza IUCN', 'Familia', 'lat', 'lon']].drop_duplicates(subset=['Nombre Científico'])
                st.dataframe(df_show, use_container_width=True)
                
                # Mapa de calor de amenazas
                st.markdown("**Focos de Amenaza:**")
                fig_heat = px.density_mapbox(
                    threatened, lat='lat', lon='lon', radius=20,
                    zoom=10, height=400, title="Concentración de Especies Amenazadas"
                )
                fig_heat.update_layout(mapbox_style="carto-positron")
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.success("✅ ¡Buenas noticias! No se encontraron especies en categorías críticas (Vulnerable, En Peligro, Crítico) en los registros disponibles.")

    else:
        st.warning("⚠️ No se encontraron registros públicos en GBIF para esta zona específica.")
        st.caption("Intenta con una cuenca más grande o verifica si hay datos disponibles en la plataforma web de GBIF.")

else:
    st.info("👈 Seleccione una zona para comenzar.")