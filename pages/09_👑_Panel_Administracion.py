# pages/09_👑_Panel_Administracion.py

import streamlit as st
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
import time
from datetime import date

# --- 1. CONFIGURACIÓN Y SEGURIDAD ---
st.set_page_config(page_title="Admin Panel", page_icon="👑", layout="wide")

ADMIN_PASSWORD = "sihcli2026" 

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🔐 Acceso Restringido")
        st.info("Panel de Control para SIHCLI-POTER (Nube)")
        pwd = st.text_input("Contraseña de Administrador:", type="password")
        if st.button("Ingresar"):
            if pwd == ADMIN_PASSWORD:
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

tab_est, tab_predios, tab_sql = st.tabs([
    "📡 Estaciones & Datos", 
    "🏡 Gestión de Predios", 
    "🛠️ Consola SQL"
])

# ==============================================================================
# TAB 1: EDITOR DE ESTACIONES Y DATOS
# ==============================================================================
with tab_est:
    st.subheader("Gestión de Estaciones Hidroclimáticas")
    
    try:
        engine = get_engine()
        # Selector Global
        df_list = pd.read_sql("SELECT id_estacion, nom_est FROM estaciones ORDER BY nom_est", engine)
        opciones = {f"{row['nom_est']} ({row['id_estacion']})": row['id_estacion'] for index, row in df_list.iterrows()}
        
        col_sel, col_dummy = st.columns([2,1])
        with col_sel:
            seleccion = st.selectbox("🔍 Seleccionar Estación:", options=list(opciones.keys()))
        
        if seleccion:
            id_sel = opciones[seleccion]
            
            # --- SUB-PESTAÑAS INTERNAS ---
            sub_meta, sub_data = st.tabs(["📝 Editar Metadatos (Ubicación)", "✏️ Corregir Datos de Lluvia"])
            
            # --- A. EDITAR METADATOS ---
            with sub_meta:
                with engine.connect() as conn:
                    df_est = pd.read_sql(text(f"SELECT * FROM estaciones WHERE id_estacion = '{id_sel}'"), conn)
                
                if not df_est.empty:
                    col_map = {c.lower().strip(): c for c in df_est.columns}
                    col_lat = col_map.get('latitude') or col_map.get('latitud')
                    col_lon = col_map.get('longitude') or col_map.get('longitud')
                    col_nom = col_map.get('nom_est')
                    col_mun = col_map.get('municipio')
                    
                    curr = df_est.iloc[0]
                    
                    with st.form("form_meta"):
                        c1, c2 = st.columns(2)
                        val_nom = curr[col_nom] if col_nom else ""
                        new_name = c1.text_input("Nombre:", value=val_nom)
                        new_muni = c2.text_input("Municipio:", value=curr[col_mun] if col_mun else "")
                        
                        val_lat = float(curr[col_lat]) if col_lat and pd.notnull(curr[col_lat]) else 0.0
                        val_lon = float(curr[col_lon]) if col_lon and pd.notnull(curr[col_lon]) else 0.0
                        
                        new_lat = c1.number_input(f"Latitud:", value=val_lat, format="%.6f")
                        new_lon = c2.number_input(f"Longitud:", value=val_lon, format="%.6f")
                        
                        if st.form_submit_button("💾 Actualizar Metadatos"):
                            if col_lat and col_lon:
                                try:
                                    sql = text(f"""
                                        UPDATE estaciones 
                                        SET {col_nom} = :nm, {col_mun} = :mu, {col_lat} = :la, {col_lon} = :lo,
                                            geometry = ST_SetSRID(ST_Point(:lo, :la), 4326)
                                        WHERE id_estacion = :id
                                    """)
                                    with engine.connect() as conn:
                                        conn.execute(sql, {"nm": new_name, "mu": new_muni, "la": new_lat, "lo": new_lon, "id": id_sel})
                                        conn.commit()
                                    st.success("✅ Metadatos actualizados.")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
            
            # --- B. CORREGIR DATOS DE LLUVIA (NUEVO) ---
            with sub_data:
                st.info(f"Corregir registros históricos para: **{seleccion}**")
                
                c_year, c_month = st.columns(2)
                sel_year = c_year.number_input("Año:", min_value=1980, max_value=2030, value=2024)
                sel_month = c_month.selectbox("Mes:", range(1, 13))
                
                # 1. Buscar dato existente
                check_sql = text("""
                    SELECT precipitation FROM precipitacion_mensual 
                    WHERE id_estacion_fk = :id 
                    AND extract(year from fecha_mes_año) = :y 
                    AND extract(month from fecha_mes_año) = :m
                """)
                
                try:
                    with engine.connect() as conn:
                        res = conn.execute(check_sql, {"id": id_sel, "y": sel_year, "m": sel_month}).fetchone()
                    
                    current_val = float(res[0]) if res else 0.0
                    msg_exist = f"Valor actual en base de datos: **{current_val} mm**" if res else "⚠️ No existe registro para esta fecha."
                    st.markdown(msg_exist)
                    
                    with st.form("form_data_rain"):
                        new_rain = st.number_input("Nuevo valor de precipitación (mm):", value=current_val, min_value=0.0)
                        
                        if st.form_submit_button("💾 Guardar Dato Corregido"):
                            # Fecha construida para el primer día del mes
                            date_str = f"{sel_year}-{sel_month:02d}-01"
                            
                            if res:
                                # UPDATE si existe
                                upd_sql = text("""
                                    UPDATE precipitacion_mensual 
                                    SET precipitation = :val 
                                    WHERE id_estacion_fk = :id 
                                    AND extract(year from fecha_mes_año) = :y 
                                    AND extract(month from fecha_mes_año) = :m
                                """)
                                with engine.connect() as conn:
                                    conn.execute(upd_sql, {"val": new_rain, "id": id_sel, "y": sel_year, "m": sel_month})
                                    conn.commit()
                                st.success(f"✅ Registro actualizado a {new_rain} mm.")
                            else:
                                # INSERT si no existe
                                ins_sql = text("""
                                    INSERT INTO precipitacion_mensual (id_estacion_fk, fecha_mes_año, precipitation)
                                    VALUES (:id, :date, :val)
                                """)
                                with engine.connect() as conn:
                                    conn.execute(ins_sql, {"id": id_sel, "date": date_str, "val": new_rain})
                                    conn.commit()
                                st.success(f"✅ Nuevo registro creado: {new_rain} mm.")
                            
                except Exception as e:
                    st.error(f"Error consultando datos: {e}")

    except Exception as e:
        st.error(f"Error de conexión: {e}")

# ==============================================================================
# TAB 2: GESTIÓN DE PREDIOS (CALIBRADO PARA TU GEOJSON)
# ==============================================================================
with tab_predios:
    st.subheader("🏡 Gestión de Predios")
    
    # A. BUSCADOR
    st.markdown("#### 🔍 Buscar y Editar")
    try:
        engine = get_engine()
        search_term = st.text_input("Buscar por 'PK_PREDIOS' (Código) o 'NOMBRE_PRE' (Nombre):", placeholder="Ej: 400200...")
        
        if search_term:
            # Buscamos en las columnas mapeadas
            query_search = text("""
                SELECT id, codigo_catastral, propietario, estado_gestion, area_ha 
                FROM predios_gestion 
                WHERE codigo_catastral ILIKE :s OR propietario ILIKE :s
                LIMIT 10
            """)
            with engine.connect() as conn:
                results = pd.read_sql(query_search, conn, params={"s": f"%{search_term}%"})
            
            if not results.empty:
                opt_p = {f"{r['propietario']} ({r['codigo_catastral']})": r['id'] for _, r in results.iterrows()}
                sel_p_id = st.selectbox("Resultados:", list(opt_p.keys()))
                
                if sel_p_id:
                    curr_p = results[results['id'] == opt_p[sel_p_id]].iloc[0]
                    st.info(f"Gestionando: **{curr_p['propietario']}**")
                    
                    with st.form("upd_predio"):
                        c1, c2 = st.columns(2)
                        new_st = c1.selectbox("Estado:", ["Identificado", "En Negociación", "Ejecutado / Conservado"], index=0)
                        if st.form_submit_button("💾 Actualizar Estado"):
                            with engine.connect() as conn:
                                conn.execute(text("UPDATE predios_gestion SET estado_gestion = :s WHERE id = :i"), {"s": new_st, "i": curr_p['id']})
                                conn.commit()
                            st.success("Estado actualizado.")
            else:
                st.warning("No encontrado. (Si acabas de subir el código, intenta Sincronizar abajo).")

    except Exception as e:
        st.error(f"Error buscador: {e}")

    st.divider()

    # B. SINCRONIZADOR (MAPEO CORREGIDO)
    with st.expander("☁️ Re-Sincronizar desde GitHub (Corrección de Columnas)", expanded=True):
        st.write("Usa esto para recargar la base de datos con las columnas correctas.")
        GITHUB_URL = "https://raw.githubusercontent.com/omejiariv/SIHCLI-POTER-2026/main/data/PrediosEjecutados.geojson"
        
        if st.button("🔄 Sincronizar Ahora", type="primary"):
            with st.status("Leyendo GeoJSON...", expanded=True) as status:
                try:
                    gdf = gpd.read_file(GITHUB_URL)
                    if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                    
                    st.write(f"Leídos {len(gdf)} registros.")
                    
                    with engine.connect() as conn:
                        conn.execute(text("TRUNCATE TABLE predios_gestion RESTART IDENTITY"))
                        
                        count = 0
                        for idx, row in gdf.iterrows():
                            # --- MAPEO BASADO EN TU IMAGEN ---
                            # PK_PREDIOS -> codigo_catastral
                            # NOMBRE_PRE (o 'NOMBRE_PRE') -> propietario
                            # AREA_HA -> area_ha
                            
                            cod = row.get('PK_PREDIOS') or row.get('pk_predios') or str(idx)
                            
                            # Intentamos varios nombres para el propietario/nombre
                            prop = row.get('NOMBRE_PRE') or row.get('NOMB_PRE') or row.get('NOMBRE_VER') or 'Sin Nombre'
                            
                            area = row.get('AREA_HA') or row.get('Shape_Area') or 0.0
                            geom_wkt = row.geometry.wkt
                            
                            sql_ins = text("""
                                INSERT INTO predios_gestion (codigo_catastral, propietario, estado_gestion, area_ha, geom)
                                VALUES (:c, :p, 'Identificado', :a, ST_Multi(ST_GeomFromText(:g, 4326)))
                            """)
                            conn.execute(sql_ins, {"c": str(cod), "p": prop, "a": float(area), "g": geom_wkt})
                            count += 1
                        conn.commit()
                        
                    status.update(label="¡Base de datos reparada!", state="complete")
                    st.success(f"✅ {count} predios cargados con los IDs correctos (PK_PREDIOS).")
                    
                except Exception as e:
                    st.error(f"Error crítico: {e}")

# ==============================================================================
# TAB 3: CONSOLA SQL
# ==============================================================================
with tab_sql:
    st.warning("Consola de administración.")
    query = st.text_area("SQL:", height=100)
    if st.button("Ejecutar"):
        if query:
            try:
                engine = get_engine()
                with engine.connect() as conn:
                    res = conn.execute(text(query))
                    conn.commit()
                    st.success("Ejecutado.")
            except Exception as e:
                st.error(f"{e}")