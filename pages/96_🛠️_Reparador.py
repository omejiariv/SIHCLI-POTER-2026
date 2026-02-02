import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from modules.db_manager import get_engine
import tempfile
import os

st.set_page_config(page_title="Reparador Blindado", layout="wide")
st.title("🛠️ Reparador de Coordenadas (Modo Seguro)")

uploaded_file = st.file_uploader("Sube el archivo mapaCVENSO.zip nuevamente:", type=["zip"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Guardar ZIP
        zip_path = os.path.join(tmp_dir, "archivo.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # 2. Leer archivo
            gdf = gpd.read_file(zip_path)
            
            # --- CORRECCIÓN DE SISTEMA DE COORDENADAS ---
            # Si el mapa viene en coordenadas planas (metros), lo pasamos a Geográficas (Lat/Lon)
            if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
                st.info(f"🔄 Convirtiendo de {gdf.crs} a EPSG:4326 (Lat/Lon)...")
                gdf = gdf.to_crs("EPSG:4326")
            
            st.success(f"✅ Archivo leído. Filas encontradas: {len(gdf)}")
            st.write("Columnas detectadas:", gdf.columns.tolist())
            
            # 3. Detectar ID y Nombre
            cols = gdf.columns
            # Buscamos ID
            c_id = next((c for c in ['Id_estacio', 'ID_ESTACIO', 'id_estacion', 'CODIGO'] if c in cols), None)
            # Buscamos Nombre
            c_nom = next((c for c in ['Nom_Est', 'NOM_EST', 'nombre', 'NOMBRE'] if c in cols), None)
            # Buscamos Altitud (opcional)
            c_alt = next((c for c in ['Altitud', 'AH', 'altitud', 'ELEV'] if c in cols), None)

            if not c_id:
                st.error("❌ No encuentro la columna del ID de la estación (Id_estacio).")
                st.stop()
                
            st.info(f"📝 Usando columna ID: '{c_id}' | Nombre: '{c_nom}'")

            # 4. Inyección a BD
            engine = get_engine()
            progress_bar = st.progress(0)
            status_text = st.empty()
            updated_count = 0
            errores = 0
            
            with engine.connect() as conn:
                trans = conn.begin()
                try:
                    total = len(gdf)
                    for i, row in gdf.iterrows():
                        try:
                            # --- EXTRACCIÓN SEGURA DE DATOS ---
                            
                            # 1. ID (Limpieza agresiva)
                            est_id = str(row[c_id]).strip()
                            
                            # 2. COORDENADAS (Desde la geometría, NO desde columnas de texto)
                            # Esto evita el error de texto en columnas numéricas
                            if row.geometry:
                                centroid = row.geometry.centroid
                                lat = centroid.y
                                lon = centroid.x
                            else:
                                continue # Si no tiene geometría, saltamos
                            
                            # 3. NOMBRE
                            nom = str(row[c_nom]).strip() if c_nom else "Estacion_" + est_id
                            
                            # 4. ALTITUD (Con manejo de error "Caribe")
                            alt = 0.0
                            if c_alt:
                                try:
                                    val = row[c_alt]
                                    # Solo convertimos si es numérico
                                    alt = float(val)
                                except:
                                    alt = 0.0 # Si falla (ej: es texto), ponemos 0
                            
                            # --- INSERT / UPDATE ---
                            stmt = text("""
                                INSERT INTO estaciones (id_estacion, nombre, latitud, longitud, altitud)
                                VALUES (:id, :nom, :lat, :lon, :alt)
                                ON CONFLICT (id_estacion) 
                                DO UPDATE SET 
                                    latitud = EXCLUDED.latitud,
                                    longitud = EXCLUDED.longitud,
                                    altitud = CASE WHEN estaciones.altitud = 0 THEN EXCLUDED.altitud ELSE estaciones.altitud END;
                            """)
                            
                            conn.execute(stmt, {
                                "id": est_id, "nom": nom, "lat": lat, "lon": lon, "alt": alt
                            })
                            updated_count += 1
                            
                        except Exception as e_row:
                            errores += 1
                            # print(f"Error en fila {i}: {e_row}")
                            continue

                        if i % 50 == 0:
                            progress_bar.progress(min(i / total, 1.0))
                            status_text.text(f"Procesando: {est_id}")
                    
                    trans.commit()
                    progress_bar.progress(1.0)
                    st.balloons()
                    st.success(f"🎉 PROCESO TERMINADO: {updated_count} estaciones actualizadas.")
                    if errores > 0:
                        st.warning(f"⚠️ Hubo {errores} filas que no se pudieron leer (probablemente datos vacíos), pero la mayoría cargó.")
                    
                    st.info("👉 Vuelve a la página 'Clima e Hidrología' y recarga. ¡Debería funcionar!")

                except Exception as e:
                    trans.rollback()
                    st.error(f"Error crítico en base de datos: {e}")

        except Exception as e:
            st.error(f"Error abriendo el ZIP: {e}")