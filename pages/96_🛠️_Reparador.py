# pages/96_🛠️_Reparador.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from modules.db_manager import get_engine
import tempfile
import os

st.set_page_config(page_title="Reparador de Coordenadas", layout="wide")
st.title("🛠️ Reparador de Coordenadas Maestro")

st.markdown("""
Esta herramienta tomará el archivo **mapaCVENSO.zip** (Shapefile), extraerá las coordenadas 
y actualizará la Base de Datos PostgreSQL para que los mapas funcionen.
""")

uploaded_file = st.file_uploader("Sube el archivo mapaCVENSO.zip aquí:", type=["zip"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Guardar y descomprimir el ZIP
        zip_path = os.path.join(tmp_dir, "archivo.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 2. Leer con GeoPandas
        try:
            gdf = gpd.read_file(zip_path)
            st.success(f"✅ Archivo leído correctamente. Se encontraron {len(gdf)} estaciones.")
            st.write("Vista previa de los datos del archivo:", gdf.head())
            
            # 3. Preparar datos para la BD
            # Mapeamos los nombres del Shapefile a los de la BD
            # Shapefile: Id_estacio, Latitud, Longitud, Nom_Est, Altitud (o AH)
            # BD: id_estacion, latitud, longitud, nombre, altitud
            
            engine = get_engine()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            updated_count = 0
            
            # Detectar nombres de columnas en el Shapefile (a veces cambian mayúsculas/minúsculas)
            cols = gdf.columns
            c_id = next((c for c in ['Id_estacio', 'ID_ESTACIO', 'id_estacion'] if c in cols), None)
            c_lat = next((c for c in ['Latitud', 'LATITUD', 'latitud'] if c in cols), None)
            c_lon = next((c for c in ['Longitud', 'LONGITUD', 'longitud'] if c in cols), None)
            c_nom = next((c for c in ['Nom_Est', 'NOM_EST', 'nombre'] if c in cols), None)
            c_alt = next((c for c in ['Altitud', 'AH', 'altitud'] if c in cols), None) # A veces es AH (Altura)

            if not c_id or not c_lat or not c_lon:
                st.error("❌ No se encontraron las columnas clave (Id, Lat, Lon) en el Shapefile.")
                st.stop()

            # 4. Inyección a la Base de Datos
            with engine.connect() as conn:
                trans = conn.begin()
                try:
                    total = len(gdf)
                    for i, row in gdf.iterrows():
                        est_id = str(row[c_id]).strip()
                        lat = float(row[c_lat])
                        lon = float(row[c_lon])
                        nom = str(row[c_nom]).strip() if c_nom else None
                        alt = float(row[c_alt]) if c_alt else 0.0
                        
                        # SQL de actualización (Upsert)
                        # Si existe, actualiza coords. Si no, crea la estación.
                        stmt = text("""
                            INSERT INTO estaciones (id_estacion, nombre, latitud, longitud, altitud)
                            VALUES (:id, :nom, :lat, :lon, :alt)
                            ON CONFLICT (id_estacion) 
                            DO UPDATE SET 
                                latitud = EXCLUDED.latitud,
                                longitud = EXCLUDED.longitud,
                                nombre = COALESCE(EXCLUDED.nombre, estaciones.nombre),
                                altitud = COALESCE(EXCLUDED.altitud, estaciones.altitud);
                        """)
                        
                        conn.execute(stmt, {
                            "id": est_id, "nom": nom, "lat": lat, "lon": lon, "alt": alt
                        })
                        
                        updated_count += 1
                        if i % 10 == 0:
                            progress = min(i / total, 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"Procesando estación {est_id}...")
                    
                    trans.commit()
                    progress_bar.progress(1.0)
                    st.balloons()
                    st.success(f"🎉 ¡ÉXITO! Se han actualizado/creado {updated_count} estaciones con coordenadas.")
                    
                    st.info("👉 Ahora ve a la página 'Clima e Hidrología' y recarga. ¡El mapa debería funcionar!")
                    
                except Exception as e:
                    trans.rollback()
                    st.error(f"Error en la actualización: {e}")
                    
        except Exception as e:
            st.error(f"Error leyendo el Shapefile: {e}")