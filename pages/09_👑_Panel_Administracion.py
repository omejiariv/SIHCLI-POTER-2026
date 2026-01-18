# pages/09_👑_Panel_Administracion.py

import streamlit as st
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
import time
from modules import admin_utils  # <--- IMPORTAMOS TU NUEVO MOTOR

# --- 1. CONFIGURACIÓN Y SEGURIDAD (MEJORADA) ---
st.set_page_config(page_title="Admin Panel", page_icon="👑", layout="wide")

def check_password():
    """Valida usuario/contraseña contra secrets.toml"""
    if st.session_state.get("password_correct", False):
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Acceso Restringido")
        st.info("Panel de Control SIHCLI-POTER (Nube)")
        
        # Validación de seguridad
        if "iri" not in st.secrets:
            st.error("⚠️ Falta configuración [iri] en secrets.toml")
            return False

        user_input = st.text_input("Usuario")
        pass_input = st.text_input("Contraseña", type="password")

        if st.button("Ingresar"):
            sec_user = st.secrets["iri"]["username"]
            sec_pass = st.secrets["iri"]["password"]

            if user_input == sec_user and pass_input == sec_pass:
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("⛔ Acceso Denegado")
    return False

if not check_password():
    st.stop()

# --- 2. CONEXIÓN ---
def get_engine():
    return create_engine(st.secrets["DATABASE_URL"])

# --- 3. INTERFAZ PRINCIPAL ---
st.title("👑 Panel de Administración y Edición de Datos")
st.markdown("---")

tab_est, tab_predios, tab_cuencas, tab_sql = st.tabs([
    "📡 Estaciones & Lluvias", 
    "🏡 Predios (Fincas)", 
    "🌊 Cuencas",
    "🛠️ Consola SQL"
])

# ==============================================================================
# TAB 1: ESTACIONES (AHORA CON CARGA AUTOMÁTICA CSV)
# ==============================================================================
with tab_est:
    st.subheader("Gestión de Estaciones Hidroclimáticas")
    
    sub_meta, sub_data, sub_upload = st.tabs([
        "📝 Editar Metadatos", 
        "✏️ Corregir Dato Individual", 
        "☁️ Carga Masiva (CSV)"
    ])
    
    engine = get_engine()
    
    # --- A. METADATOS (Conservado) ---
    with sub_meta:
        df_list = pd.read_sql("SELECT id_estacion, nom_est FROM estaciones ORDER BY nom_est", engine)
        opciones = {f"{row['nom_est']} ({row['id_estacion']})": row['id_estacion'] for index, row in df_list.iterrows()}
        seleccion = st.selectbox("🔍 Seleccionar Estación:", options=list(opciones.keys()), key="sel_meta")
        
        if seleccion:
            id_sel = opciones[seleccion]
            with engine.connect() as conn:
                df_est = pd.read_sql(text(f"SELECT * FROM estaciones WHERE id_estacion = '{id_sel}'"), conn)
            
            if not df_est.empty:
                col_map = {c.lower().strip(): c for c in df_est.columns}
                col_lat = col_map.get('latitude') or col_map.get('latitud')
                col_lon = col_map.get('longitude') or col_map.get('longitud')
                curr = df_est.iloc[0]
                
                with st.form("form_meta"):
                    c1, c2 = st.columns(2)
                    new_name = c1.text_input("Nombre:", value=curr[col_map.get('nom_est')] if col_map.get('nom_est') else "")
                    new_muni = c2.text_input("Municipio:", value=curr[col_map.get('municipio')] if col_map.get('municipio') else "")
                    
                    val_lat = float(curr[col_lat]) if col_lat and pd.notnull(curr[col_lat]) else 0.0
                    val_lon = float(curr[col_lon]) if col_lon and pd.notnull(curr[col_lon]) else 0.0
                    new_lat = c1.number_input("Latitud:", value=val_lat, format="%.6f")
                    new_lon = c2.number_input("Longitud:", value=val_lon, format="%.6f")
                    
                    if st.form_submit_button("💾 Actualizar"):
                        if col_lat and col_lon:
                            sql = text(f"""
                                UPDATE estaciones SET {col_map.get('nom_est')} = :n, {col_map.get('municipio')} = :m, 
                                {col_lat} = :la, {col_lon} = :lo, geometry = ST_SetSRID(ST_Point(:lo, :la), 4326)
                                WHERE id_estacion = :id
                            """)
                            with engine.connect() as conn:
                                conn.execute(sql, {"n": new_name, "m": new_muni, "la": new_lat, "lo": new_lon, "id": id_sel})
                                conn.commit()
                            st.success("✅ Guardado.")

    # --- B. DATO INDIVIDUAL (Conservado) ---
    with sub_data:
        st.info("Corrección puntual de datos históricos.")
        if seleccion:
            id_sel_data = opciones[seleccion]
            c_y, c_m = st.columns(2)
            sel_year = c_y.number_input("Año:", 1980, 2030, 2025)
            sel_month = c_m.selectbox("Mes:", range(1,13))
            
            with engine.connect() as conn:
                res = conn.execute(text("SELECT precipitation FROM precipitacion_mensual WHERE id_estacion_fk=:id AND extract(year from fecha_mes_año)=:y AND extract(month from fecha_mes_año)=:m"), {"id": id_sel_data, "y": sel_year, "m": sel_month}).fetchone()
            
            curr_val = float(res[0]) if (res and res[0] is not None) else 0.0
            st.write(f"Valor actual: **{curr_val} mm**")
            
            with st.form("fix_data"):
                 new_val = st.number_input("Nuevo Valor:", value=curr_val)
                 if st.form_submit_button("💾 Corregir"):
                     date_str = f"{sel_year}-{sel_month:02d}-01"
                     with engine.connect() as conn:
                         # Upsert manual
                         if res:
                             conn.execute(text("UPDATE precipitacion_mensual SET precipitation=:v WHERE id_estacion_fk=:id AND extract(year from fecha_mes_año)=:y AND extract(month from fecha_mes_año)=:m"), {"v": new_val, "id": id_sel_data, "y": sel_year, "m": sel_month})
                         else:
                             conn.execute(text("INSERT INTO precipitacion_mensual (id_estacion_fk, fecha_mes_año, precipitation) VALUES (:id, :d, :v)"), {"id": id_sel_data, "d": date_str, "v": new_val})
                         conn.commit()
                     st.success("✅ Corregido.")

    # --- C. CARGA MASIVA (MEJORADA CON ADMIN_UTILS) ---
    with sub_upload:
        st.markdown("#### ☁️ Carga Masiva Automática")
        st.info("Sube el archivo Excel/CSV 'ancho' (con meses en columnas). El sistema lo convertirá y subirá a la base de datos.")
        
        uploaded_file = st.file_uploader("Arrastra tu archivo CSV aquí (separado por ;)", type=["csv"])
        
        if uploaded_file:
            # 1. Procesamiento (Usando tu nuevo módulo)
            if st.button("🚀 1. Analizar Archivo"):
                with st.spinner("Transformando formato ancho a largo..."):
                    df_procesado, error = admin_utils.procesar_archivo_precipitacion(uploaded_file)
                    
                    if error:
                        st.error(f"❌ Error: {error}")
                    else:
                        st.session_state['df_upload_ready'] = df_procesado
                        st.success(f"✅ Archivo válido. Se encontraron {len(df_procesado)} datos de lluvia.")
                        st.dataframe(df_procesado.head())

            # 2. Inserción a Base de Datos
            if 'df_upload_ready' in st.session_state:
                st.markdown("---")
                st.warning(f"⚠️ Estás a punto de subir {len(st.session_state['df_upload_ready'])} registros a la base de datos PRO.")
                
                if st.button("💾 2. Confirmar y Subir a Supabase"):
                    df_final = st.session_state['df_upload_ready']
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        with engine.connect() as conn:
                            sql_insert = text("""
                                INSERT INTO precipitacion_mensual (id_estacion_fk, fecha_mes_año, precipitation)
                                VALUES (:id, :dt, :val)
                                ON CONFLICT (id_estacion_fk, fecha_mes_año) 
                                DO UPDATE SET precipitation = EXCLUDED.precipitation
                            """)
                            
                            data_to_insert = df_final.to_dict(orient='records')
                            total = len(data_to_insert)
                            batch_size = 500
                            
                            for i in range(0, total, batch_size):
                                batch = data_to_insert[i:i+batch_size]
                                batch_mapped = [{"id": r['id_estacion'], "dt": r['fecha_mes_año'], "val": r['precipitation']} for r in batch]
                                conn.execute(sql_insert, batch_mapped)
                                conn.commit()
                                
                                prog = min((i + batch_size) / total, 1.0)
                                progress_bar.progress(prog)
                                status_text.text(f"Subiendo lote {i} de {total}...")
                                
                        st.success("🎉 ¡Carga Masiva Completada con Éxito!")
                        del st.session_state['df_upload_ready']
                        
                    except Exception as e:
                        st.error(f"Error subiendo a base de datos: {e}")

# ==============================================================================
# TAB 2: GESTIÓN DE PREDIOS (TU CÓDIGO ORIGINAL CONSERVADO)
# ==============================================================================
with tab_predios:
    st.subheader("🏡 Gestión de Predios")
    st.markdown("#### 🔍 Buscar y Editar")
    try:
        engine = get_engine()
        search_term = st.text_input("Buscar (Código, Nombre Finca o Propietario):", placeholder="Ej: La Esperanza...")
        
        if search_term:
            query_search = text("""
                SELECT id, codigo_catastral, nombre_predio, propietario, vereda, municipio, estado_gestion, area_ha 
                FROM predios_gestion 
                WHERE codigo_catastral ILIKE :s OR nombre_predio ILIKE :s OR propietario ILIKE :s
                LIMIT 10
            """)
            with engine.connect() as conn:
                results = pd.read_sql(query_search, conn, params={"s": f"%{search_term.strip()}%"})
            
            if not results.empty:
                opt_p = {f"{r['nombre_predio']} - {r['propietario']} ({r['codigo_catastral']})": r['id'] for _, r in results.iterrows()}
                sel_p_id = st.selectbox("Resultados:", list(opt_p.keys()))
                
                if sel_p_id:
                    curr_p = results[results['id'] == opt_p[sel_p_id]].iloc[0]
                    with st.form("upd_predio"):
                        c1, c2 = st.columns(2)
                        new_nom = c1.text_input("Nombre Finca:", value=curr_p['nombre_predio'])
                        new_prop = c2.text_input("Propietario:", value=curr_p['propietario'])
                        c3, c4 = st.columns(2)
                        new_ver = c3.text_input("Vereda:", value=curr_p['vereda'])
                        new_mun = c4.text_input("Municipio:", value=curr_p['municipio'])
                        new_st = st.selectbox("Estado:", ["Identificado", "En Negociación", "Ejecutado / Conservado", "Descartado"], index=0)
                        
                        if st.form_submit_button("💾 Guardar Cambios"):
                            with engine.connect() as conn:
                                sql = text("UPDATE predios_gestion SET nombre_predio=:n, propietario=:p, vereda=:v, municipio=:m, estado_gestion=:s WHERE id=:i")
                                conn.execute(sql, {"n": new_nom, "p": new_prop, "v": new_ver, "m": new_mun, "s": new_st, "i": curr_p['id']})
                                conn.commit()
                            st.success("✅ Actualizado.")
                            time.sleep(1); st.rerun()
            else: st.warning("No encontrado.")
            
    except Exception as e: st.error(f"Error: {e}")
    
    with st.expander("☁️ Re-Sincronizar Predios (GitHub)", expanded=False):
        if st.button("🔄 Sincronizar Predios", type="primary"):
            with st.status("Procesando...", expanded=True):
                try:
                    url = "https://raw.githubusercontent.com/omejiariv/SIHCLI-POTER-2026/main/data/PrediosEjecutados.geojson"
                    gdf = gpd.read_file(url)
                    if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                    with engine.connect() as conn:
                        conn.execute(text("TRUNCATE TABLE predios_gestion RESTART IDENTITY"))
                        for idx, row in gdf.iterrows():
                            cod = str(row.get('PK_PREDIOS') or "")
                            nom_pre = row.get('NOMBRE_PRE') or 'Sin Nombre'
                            vereda = row.get('NOMBRE_VER') or 'Sin Vereda'
                            muni = row.get('NOMB_MPIO') or row.get('NOM_MPIO') or 'Sin Municipio'
                            area = row.get('AREA_HA') or row.get('Shape_Area') or 0.0
                            sql = text("INSERT INTO predios_gestion (codigo_catastral, nombre_predio, propietario, vereda, municipio, estado_gestion, area_ha, geom) VALUES (:c, :np, 'Por Asignar', :v, :m, 'Identificado', :a, ST_Multi(ST_GeomFromText(:g, 4326)))")
                            conn.execute(sql, {"c": cod, "np": nom_pre, "v": vereda, "m": muni, "a": float(area), "g": row.geometry.wkt})
                        conn.commit()
                    st.success("✅ Sincronización completa.")
                except Exception as e: st.error(f"Error: {e}")

# ==============================================================================
# TAB 3: CUENCAS (TU CÓDIGO ORIGINAL CONSERVADO)
# ==============================================================================
with tab_cuencas:
    st.subheader("🌊 Gestión de Cuencas")
    try:
        engine = get_engine()
        search_c = st.text_input("Buscar Cuenca:", placeholder="Ej: Rio Grande...")
        if search_c:
            q_c = text("SELECT * FROM cuencas_gestion WHERE nombre ILIKE :s LIMIT 10")
            with engine.connect() as conn:
                res_c = pd.read_sql(q_c, conn, params={"s": f"%{search_c}%"})
            if not res_c.empty:
                opt_c = {f"{r['nombre']} ({r['codigo']})": r['id'] for _, r in res_c.iterrows()}
                sel_c_id = st.selectbox("Cuencas:", list(opt_c.keys()))
                if sel_c_id:
                    curr_c = res_c[res_c['id'] == opt_c[sel_c_id]].iloc[0]
                    with st.form("upd_cuenca"):
                        new_name_c = st.text_input("Nombre:", value=curr_c['nombre'])
                        new_obs = st.text_area("Observación:", value=curr_c.get('observacion', ''))
                        if st.form_submit_button("💾 Guardar"):
                            with engine.connect() as conn:
                                conn.execute(text("UPDATE cuencas_gestion SET nombre=:n, observacion=:o WHERE id=:i"), {"n": new_name_c, "o": new_obs, "i": curr_c['id']})
                                conn.commit()
                            st.success("Actualizado."); time.sleep(1); st.rerun()
    except Exception as e: st.error(f"Error: {e}")

# ==============================================================================
# TAB 4: CONSOLA SQL (TU CÓDIGO ORIGINAL CONSERVADO)
# ==============================================================================
with tab_sql:
    st.warning("⚠️ Consola SQL - Uso Avanzado")
    q = st.text_area("SQL:")
    if st.button("Ejecutar"):
        if q:
            try:
                engine = get_engine()
                with engine.connect() as conn:
                    if q.lower().strip().startswith("select"): 
                        st.dataframe(pd.read_sql(text(q), conn))
                    else: 
                        res = conn.execute(text(q))
                        conn.commit()
                        st.success("Hecho.")
            except Exception as e: st.error(str(e))