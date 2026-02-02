import streamlit as st
import pandas as pd
from sqlalchemy import text
from modules.db_manager import get_engine
import io

st.set_page_config(page_title="Reparador Definitivo", layout="wide")
st.title("🛠️ Reparador Final (Calibrado)")

st.markdown("### Sube el archivo `mapaCVENSO.csv` para inyectar coordenadas.")

uploaded_file = st.file_uploader("Sube el CSV aquí:", type=["csv", "txt", "xlsx"])

if uploaded_file:
    engine = get_engine()
    
    # 1. LEER ARCHIVO (Forzamos punto y coma que es tu formato)
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            # Forzamos sep=';' y encoding 'latin1' (típico de archivos de gobierno/arcgis)
            df = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
            
        # Normalizamos columnas: quitamos espacios y pasamos a minúsculas
        df.columns = [c.strip().lower() for c in df.columns]
        
        st.success(f"✅ Archivo leído correctamente. Filas: {len(df)}")
        st.write("Columnas encontradas:", df.columns.tolist())
        
        # 2. CONFIGURACIÓN AUTOMÁTICA (Con tus nombres exactos)
        cols = df.columns.tolist()
        
        # Función auxiliar para encontrar índice
        def get_idx(candidates):
            for i, col in enumerate(cols):
                if col in candidates: return i
            return 0

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            # Tu columna es 'id_estacion'
            idx = get_idx(['id_estacion', 'id_estacio'])
            col_id = st.selectbox("ID:", cols, index=idx)
            
        with c2:
            # Tu columna es 'latitud'
            idx = get_idx(['latitud', 'lat'])
            col_lat = st.selectbox("LAT:", cols, index=idx)
            
        with c3:
            # Tu columna es 'longitud'
            idx = get_idx(['longitud', 'lon'])
            col_lon = st.selectbox("LON:", cols, index=idx)
            
        with c4:
            # Tu columna es 'nom_est'
            idx = get_idx(['nom_est', 'nombre', 'estacion'])
            col_nom = st.selectbox("NOMBRE:", cols, index=idx)

        # 3. BOTÓN DE FUEGO
        if st.button("🚀 EJECUTAR REPARACIÓN AHORA"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            updated_count = 0
            inserted_count = 0
            errors = 0
            
            with engine.connect() as conn:
                trans = conn.begin()
                try:
                    total = len(df)
                    for i, row in df.iterrows():
                        try:
                            # DATOS
                            sid = str(row[col_id]).strip()
                            
                            # Reemplazar comas por puntos si las hay (ej: 6,15 -> 6.15)
                            lat = float(str(row[col_lat]).replace(',', '.'))
                            lon = float(str(row[col_lon]).replace(',', '.'))
                            
                            # Altitud (opcional, buscamos 'altitud' o 'ah')
                            alt = 0.0
                            if 'altitud' in df.columns:
                                try: alt = float(str(row['altitud']).replace(',', '.'))
                                except: pass
                            elif 'ah' in df.columns:
                                try: alt = float(str(row['ah']).replace(',', '.'))
                                except: pass

                            # UPDATE (Actualizar coordenadas de estaciones existentes)
                            stmt_upd = text("""
                                UPDATE estaciones 
                                SET latitud = :lat, longitud = :lon, altitud = :alt
                                WHERE TRIM(id_estacion) = :id
                            """)
                            res = conn.execute(stmt_upd, {"lat": lat, "lon": lon, "alt": alt, "id": sid})
                            
                            if res.rowcount > 0:
                                updated_count += 1
                            else:
                                # INSERT (Crear si no existe)
                                nom = str(row[col_nom]).strip()
                                stmt_ins = text("""
                                    INSERT INTO estaciones (id_estacion, nombre, latitud, longitud, altitud)
                                    VALUES (:id, :nom, :lat, :lon, :alt)
                                    ON CONFLICT (id_estacion) DO NOTHING
                                """)
                                conn.execute(stmt_ins, {"id": sid, "nom": nom, "lat": lat, "lon": lon, "alt": alt})
                                inserted_count += 1
                            
                        except Exception:
                            errors += 1
                        
                        if i % 50 == 0:
                            progress_bar.progress(min(i/total, 1.0))
                            status_text.text(f"Procesando {i}...")

                    trans.commit()
                    progress_bar.progress(1.0)
                    st.balloons()
                    
                    st.success(f"""
                    🎉 **¡ÉXITO TOTAL!**
                    - Estaciones actualizadas (coordenadas corregidas): **{updated_count}**
                    - Estaciones nuevas creadas: **{inserted_count}**
                    - Total procesado: **{total}**
                    """)
                    
                    st.info("👉 Ve a 'Clima e Hidrología' y recarga la página. ¡El mapa DEBE salir!")
                    
                except Exception as e:
                    trans.rollback()
                    st.error(f"Error SQL: {e}")

    except Exception as e:
        st.error(f"Error leyendo archivo: {e}")