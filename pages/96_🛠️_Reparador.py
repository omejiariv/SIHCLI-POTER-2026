import streamlit as st
import pandas as pd
from sqlalchemy import text
from modules.db_manager import get_engine
import io

st.set_page_config(page_title="Reparador Final", layout="wide")
st.title("🛠️ Reparador de Coordenadas (Versión Definitiva)")

st.markdown("""
### Instrucciones:
1. Sube el archivo **mapaCVENSO.csv** (el que tiene separador `;` y minúsculas).
2. Verifica que los selectores coincidan automáticamente.
3. Dale al botón rojo.
""")

uploaded_file = st.file_uploader("Sube el archivo CSV aquí:", type=["csv", "txt", "xlsx"])

if uploaded_file:
    engine = get_engine()
    
    # 1. LEER ARCHIVO (Prioridad: Punto y coma ';')
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            # Intentamos primero con punto y coma (Tu formato)
            try:
                df = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
                if len(df.columns) < 2: # Si falló, probamos coma
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=',', encoding='latin1')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
            
        st.success(f"✅ Archivo cargado. Filas: {len(df)}")
        st.write("Vista previa:", df.head(3))
        
        # Normalizar nombres de columnas del DF (todo a minúsculas y sin espacios)
        df.columns = [c.strip().lower() for c in df.columns]
        cols = df.columns.tolist()
        
        # 2. SELECTORES INTELIGENTES (Ajustados a tus nombres)
        st.divider()
        st.subheader("🔗 Mapeo de Columnas")
        
        def get_index(options, candidates):
            for i, opt in enumerate(options):
                if opt in candidates: return i # Coincidencia exacta
                if any(c in opt for c in candidates): return i # Coincidencia parcial
            return 0

        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            # Buscamos 'id_estacion' o 'id_estacio'
            col_id = st.selectbox("ID ESTACIÓN:", cols, index=get_index(cols, ['id_estacion', 'id_estacio']))
        with c2:
            # Buscamos 'latitud'
            col_lat = st.selectbox("LATITUD:", cols, index=get_index(cols, ['latitud']))
        with c3:
            # Buscamos 'longitud'
            col_lon = st.selectbox("LONGITUD:", cols, index=get_index(cols, ['longitud']))
        with c4:
            # Buscamos 'nom_est' o 'nombre'
            col_nom = st.selectbox("NOMBRE:", ["(Ninguna)"] + cols, index=get_index(["(Ninguna)"] + cols, ['nom_est', 'nombre']) )

        # 3. EJECUCIÓN
        if st.button("🚀 REPARAR BASE DE DATOS AHORA"):
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
                            # Datos Crudos
                            sid = str(row[col_id]).strip()
                            raw_lat = str(row[col_lat]).replace(',', '.')
                            raw_lon = str(row[col_lon]).replace(',', '.')
                            
                            # Conversión
                            lat = float(raw_lat)
                            lon = float(raw_lon)
                            
                            # Corrección de Altitud si existe
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
                                # INSERT (Si no existe)
                                nom = f"Estación {sid}"
                                if col_nom != "(Ninguna)":
                                    nom = str(row[col_nom]).strip()
                                
                                stmt_ins = text("""
                                    INSERT INTO estaciones (id_estacion, nombre, latitud, longitud, altitud)
                                    VALUES (:id, :nom, :lat, :lon, :alt)
                                    ON CONFLICT (id_estacion) DO UPDATE 
                                    SET latitud=EXCLUDED.latitud, longitud=EXCLUDED.longitud
                                """)
                                conn.execute(stmt_ins, {"id": sid, "nom": nom, "lat": lat, "lon": lon, "alt": alt})
                                inserted_count += 1
                            
                        except Exception:
                            errors += 1
                            
                        if i % 50 == 0:
                            progress_bar.progress(min(i / total, 1.0))
                            status_text.text(f"Procesando {i}/{total}...")
                    
                    trans.commit()
                    progress_bar.progress(1.0)
                    st.balloons()
                    
                    st.success(f"""
                    🎉 **PROCESO FINALIZADO**
                    - Filas en archivo: {total}
                    - Actualizadas en BD: {updated_count}
                    - Insertadas nuevas: {inserted_count}
                    """)
                    
                    st.info("👉 VE AHORA A 'CLIMA E HIDROLOGÍA'.")
                    
                except Exception as e:
                    trans.rollback()
                    st.error(f"Error crítico: {e}")

    except Exception as e:
        st.error(f"Error leyendo el archivo: {e}")