import streamlit as st
import pandas as pd
from sqlalchemy import text
from modules.db_manager import get_engine
import io

st.set_page_config(page_title="Reparador CSV", layout="wide")
st.title("🛠️ Reparador Masivo con CSV")

st.markdown("""
### Instrucciones:
1. Sube tu archivo **mapaCVENSO.csv** (o Excel .xlsx).
2. El sistema buscará las columnas de Latitud y Longitud.
3. Actualizará las 790 estaciones de la base de datos.
""")

uploaded_file = st.file_uploader("Sube el archivo CSV o Excel aquí:", type=["csv", "xlsx", "txt"])

if uploaded_file:
    engine = get_engine()
    
    # 1. LEER ARCHIVO (Inteligencia para detectar formato)
    try:
        if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
            # Probamos separadores comunes
            try:
                df = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
                if len(df.columns) < 2: 
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=',', encoding='latin1')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ Archivo cargado. Filas encontradas: {len(df)}")
        st.write("Primeras 3 filas:", df.head(3))
        
        # 2. LIMPIEZA DE NOMBRES DE COLUMNAS
        # Quitamos espacios y pasamos a minúsculas para buscar mejor
        df.columns = [c.strip() for c in df.columns]
        
        # 3. DETECTAR COLUMNAS
        cols = df.columns.tolist()
        
        # ID
        c_id = next((c for c in cols if c.lower() in ['id_estacion', 'id_estacio', 'codigo', 'cod']), None)
        # Latitud
        c_lat = next((c for c in cols if c.lower() in ['latitud', 'lat', 'latitud_geo', 'y']), None)
        # Longitud
        c_lon = next((c for c in cols if c.lower() in ['longitud', 'lon', 'longitud_geo', 'x']), None)
        # Altitud (Opcional)
        c_alt = next((c for c in cols if c.lower() in ['altitud', 'alt', 'elevacion', 'z', 'ah']), None)
        
        st.info(f"📍 Columnas detectadas -> ID: `{c_id}` | Lat: `{c_lat}` | Lon: `{c_lon}`")
        
        if not c_id or not c_lat or not c_lon:
            st.error("❌ No pude identificar las columnas. Asegúrate que el CSV tenga encabezados como 'Id_estacion', 'Latitud', 'Longitud'.")
            st.stop()
            
        # 4. BOTÓN DE ACCIÓN
        if st.button("🚀 INICIAR REPARACIÓN MASIVA"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            updated_count = 0
            errors = 0
            
            with engine.connect() as conn:
                trans = conn.begin()
                try:
                    total = len(df)
                    for i, row in df.iterrows():
                        try:
                            # Limpieza de datos (Manejo de comas decimales '5,4')
                            sid = str(row[c_id]).strip()
                            
                            raw_lat = str(row[c_lat]).replace(',', '.')
                            raw_lon = str(row[c_lon]).replace(',', '.')
                            
                            lat = float(raw_lat)
                            lon = float(raw_lon)
                            
                            alt = 0.0
                            if c_alt:
                                try:
                                    alt = float(str(row[c_alt]).replace(',', '.'))
                                except: pass
                                
                            # SQL UPDATE
                            # Solo actualizamos coordenadas donde el ID coincida
                            stmt = text("""
                                UPDATE estaciones 
                                SET latitud = :lat, longitud = :lon, altitud = :alt
                                WHERE id_estacion = :id
                            """)
                            
                            result = conn.execute(stmt, {"lat": lat, "lon": lon, "alt": alt, "id": sid})
                            
                            # Si no actualizó nada (porque el ID no existía), lo insertamos
                            if result.rowcount == 0:
                                # Recuperamos nombre si existe, sino genérico
                                c_nom = next((c for c in cols if c.lower() in ['nombre', 'nom_est']), None)
                                nom = str(row[c_nom]).strip() if c_nom else f"Est {sid}"
                                
                                stmt_ins = text("""
                                    INSERT INTO estaciones (id_estacion, nombre, latitud, longitud, altitud)
                                    VALUES (:id, :nom, :lat, :lon, :alt)
                                """)
                                conn.execute(stmt_ins, {"id": sid, "nom": nom, "lat": lat, "lon": lon, "alt": alt})
                            
                            updated_count += 1
                            
                        except Exception as e:
                            # print(f"Error fila {i}: {e}")
                            errors += 1
                            
                        if i % 50 == 0:
                            progress_bar.progress(min(i / total, 1.0))
                            status_text.text(f"Procesando {i}/{total}...")
                    
                    trans.commit()
                    progress_bar.progress(1.0)
                    st.balloons()
                    st.success(f"🎉 ¡HECHO! Se procesaron {updated_count} filas.")
                    if errors > 0:
                        st.warning(f"Hubo {errors} filas con errores de formato numérico.")
                        
                    st.info("👉 AHORA SÍ: Ve a 'Clima e Hidrología' y deberías ver el mapa lleno de puntos.")
                    
                except Exception as e:
                    trans.rollback()
                    st.error(f"Error en BD: {e}")

    except Exception as e:
        st.error(f"Error leyendo el archivo: {e}")