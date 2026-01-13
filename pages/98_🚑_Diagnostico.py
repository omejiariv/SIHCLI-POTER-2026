import streamlit as st
from modules import data_processor
import geopandas as gpd
import pandas as pd

st.set_page_config(page_title="Mapeo de Datos", page_icon="🗺️")
st.title("🗺️ Mapeo de Índices de data_processor")

try:
    with st.spinner("Cargando datos..."):
        # Limpiamos caché para ver la realidad
        data_processor.load_and_process_all_data.clear()
        all_data = data_processor.load_and_process_all_data()

    st.success(f"📦 La función devolvió {len(all_data)} elementos.")
    
    for i, item in enumerate(all_data):
        st.divider()
        st.subheader(f"Índice [{i}]")
        st.write(f"Tipo: `{type(item)}`")
        
        if hasattr(item, 'columns'):
            st.write(f"Columnas ({len(item.columns)}):", item.columns.tolist())
            st.write(f"Filas: {len(item)}")
            
            # Pista para identificar
            if 'geometry' in item.columns:
                st.info("📍 Es un GeoDataFrame (Mapa)")
                if 'cuenca' in str(item.columns).lower():
                    st.success("🌟 ¡ESTAS PARECEN LAS CUENCAS!")
            elif 'precipitation' in item.columns:
                st.info("💧 Parecen datos de lluvia")
            elif 'nom_est' in item.columns and len(item.columns) < 15:
                st.info("🏠 Parecen las Estaciones")
        else:
            st.warning("No es un DataFrame")

except Exception as e:
    st.error(f"Error: {e}")