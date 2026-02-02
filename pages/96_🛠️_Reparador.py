import streamlit as st
import pandas as pd
from sqlalchemy import text
from modules.db_manager import get_engine

st.set_page_config(page_title="Reparador Anti-BOM", layout="wide")
st.title("🛠️ Reparador de Coordenadas (Limpieza de Caracteres Especiales)")

st.markdown("""
### El problema detectado:
Tu archivo tiene una firma oculta (`ï»¿`) en el nombre de la columna.
Este script la elimina antes de procesar.

### Instrucciones:
1. Sube el archivo **mapaCVENSO.csv**.
2. Verifica que los selectores se pongan verdes.
3. Dale al botón rojo.
""")

uploaded_file = st.file_uploader("Sube el archivo CSV aquí:", type=["csv", "txt", "xlsx"])

if uploaded_file:
    engine = get_engine()
    
    # 1. LEER ARCHIVO CON LIMPIEZA DE BOM (ï»¿)
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            # INTENTO 1: UTF-8-SIG (Esto elimina el BOM automáticamente)
            try:
                df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
            except:
                # Fallback: Latin1
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
            
            # Si falló la separación por punto y coma, intentamos coma
            if len(df.columns) < 2:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8-sig')

        # 2. LIMPIEZA QUIRÚRGICA DE COLUMNAS
        # Eliminamos explícitamente los caracteres basura del nombre de la columna
        new_cols = []
        for c in df.columns:
            clean_c = str(c).strip().lower()
            clean_c = clean_c.replace('ï»¿', '') # Eliminar BOM visible
            clean_c = clean_c.replace('\ufeff', '') # Eliminar BOM invisible
            clean_c = clean_c.replace('"', '').replace("'", "") # Eliminar comillas
            new_cols.append(clean_c)
            
        df.columns = new_cols
        
        st.success(f"✅ Archivo leído y limpiado. Filas: {len(df)}")
        st.write("Columnas LIMPIAS encontradas:", df.columns.tolist())
        
        # 3. SELECTORES INTELIGENTES
        cols = df.columns.tolist()
        
        def get_index(options, candidates):
            for i, opt in enumerate(options):
                if opt in candidates: return i 
                if any(c in opt for c in candidates): return i
            return 0

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            idx = get_index(cols, ['id_estacion', 'id_estacio', 'codigo'])
            col_id = st.selectbox("ID ESTACIÓN:", cols, index=idx)
        with c2:
            idx = get_index(cols, ['latitud', 'lat'])
            col_lat = st.selectbox("LATITUD:", cols, index=idx)
        with c3:
            idx = get_index(cols, ['longitud', 'lon'])
            col_lon = st.selectbox("LONGITUD:", cols, index=idx)
        with c4:
            idx = get_index(cols, ['nom_est', 'nombre', 'estacion'])
            col_nom = st.selectbox("NOMBRE:", cols, index=idx)

        # 4. EJECUCIÓN
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
                            
                            # Manejo de decimales (coma por punto)
                            raw_lat = str(row[col_lat]).replace(',', '.')
                            raw_lon = str(row[col_lon]).replace(',', '.')
                            
                            lat = float(raw_lat)
                            lon = float(raw_lon)
                            
                            # Altitud
                            alt = 0.0
                            if 'altitud' in df.columns:
                                try: alt = float(str(row['altitud']).replace(',', '.'))
                                except: pass
                            elif 'ah' in df.columns:
                                try: alt = float(str(row['ah']).replace(',', '.'))
                                except: pass

                            # UPDATE
                            stmt_upd = text("""
                                UPDATE estaciones 
                                SET latitud = :lat, longitud = :lon, altitud = :alt
                                WHERE TRIM(id_estacion) = :id
                            """)
                            res = conn.execute(stmt_upd, {"lat": lat, "lon": lon, "alt": alt, "id": sid})
                            
                            if res.rowcount > 0:
                                updated_count += 1
                            else:
                                # INSERT
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
                            status_text.text(f"Procesando {i}/{total}...")

                    trans.commit()
                    progress_bar.progress(1.0)
                    st.balloons()
                    
                    st.success(f"""
                    🎉 **¡ÉXITO TOTAL!**
                    - Estaciones con coordenadas actualizadas: **{updated_count}**
                    - Estaciones nuevas: **{inserted_count}**
                    """)
                    
                    st.info("👉 VE AHORA A 'CLIMA E HIDROLOGÍA'. EL MAPA DEBE FUNCIONAR.")
                    
                except Exception as e:
                    trans.rollback()
                    st.error(f"Error SQL: {e}")

    except Exception as e:
        st.error(f"Error leyendo archivo: {e}")