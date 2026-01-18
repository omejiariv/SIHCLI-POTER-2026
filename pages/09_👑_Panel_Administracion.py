# pages/09_👑_Panel_Administracion.py

import streamlit as st
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
import time

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
    "📡 Estaciones (Editor)", 
    "🏡 Predios (Gestión)", 
    "🛠️ Consola SQL"
])

# ==============================================================================
# TAB 1: EDITOR DE ESTACIONES
# ==============================================================================
with tab_est:
    st.subheader("Modificar Metadatos de Estaciones")
    try:
        engine = get_engine()
        df_list = pd.read_sql("SELECT id_estacion, nom_est FROM estaciones ORDER BY nom_est", engine)
        opciones = {f"{row['nom_est']} ({row['id_estacion']})": row['id_estacion'] for index, row in df_list.iterrows()}
        
        seleccion = st.selectbox("🔍 Buscar Estación:", options=list(opciones.keys()))
        
        if seleccion:
            id_sel = opciones[seleccion]
            with engine.connect() as conn:
                df_est = pd.read_sql(text(f"SELECT * FROM estaciones WHERE id_estacion = '{id_sel}'"), conn)
            
            if not df_est.empty:
                col_map = {c.lower().strip(): c for c in df_est.columns}
                col_lat = col_map.get('latitude') or col_map.get('latitud')
                col_lon = col_map.get('longitude') or col_map.get('longitud')
                col_nom = col_map.get('nom_est')
                col_mun = col_map.get('municipio')
                
                curr = df_est.iloc[0]
                
                with st.form("form_estacion"):
                    c1, c2 = st.columns(2)
                    val_nom = curr[col_nom] if col_nom else ""
                    val_mun = curr[col_mun] if col_mun else ""
                    val_lat = float(curr[col_lat]) if col_lat and pd.notnull(curr[col_lat]) else 0.0
                    val_lon = float(curr[col_lon]) if col_lon and pd.notnull(curr[col_lon]) else 0.0
                    
                    new_name = c1.text_input("Nombre:", value=val_nom)
                    new_muni = c2.text_input("Municipio:", value=val_mun)
                    new_lat = c1.number_input(f"Latitud ({col_lat}):", value=val_lat, format="%.6f")
                    new_lon = c2.number_input(f"Longitud ({col_lon}):", value=val_lon, format="%.6f")
                    
                    st.caption("ℹ️ Al guardar, se actualizarán las coordenadas y la geometría espacial.")
                    
                    if st.form_submit_button("💾 Guardar Cambios"):
                        if col_lat and col_lon:
                            try:
                                sql = text(f"""
                                    UPDATE estaciones 
                                    SET {col_nom} = :nm, {col_mun} = :mu, 
                                        {col_lat} = :la, {col_lon} = :lo,
                                        geometry = ST_SetSRID(ST_Point(:lo, :la), 4326)
                                    WHERE id_estacion = :id
                                """)
                                with engine.connect() as conn:
                                    conn.execute(sql, {"nm": new_name, "mu": new_muni, "la": new_lat, "lo": new_lon, "id": id_sel})
                                    conn.commit()
                                st.success("✅ Estación actualizada exitosamente.")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error SQL: {e}")
                        else:
                            st.error("No se encontraron las columnas de coordenadas.")
    except Exception as e:
        st.error(f"Error: {e}")

# ==============================================================================
# TAB 2: GESTIÓN DE PREDIOS (¡NUEVO: EDICIÓN INDIVIDUAL!)
# ==============================================================================
with tab_predios:
    st.subheader("🏡 Gestión de Predios")
    
    # SECCIÓN A: EDICIÓN INDIVIDUAL (LO NUEVO)
    st.markdown("#### ✏️ Actualizar Estado de Predio")
    
    try:
        engine = get_engine()
        # Buscador
        search_term = st.text_input("🔍 Buscar por Propietario o Código Catastral:", placeholder="Ej: Perez, o 001-...")
        
        if search_term:
            # Búsqueda flexible (ILIKE es insensible a mayúsculas)
            query_search = text("""
                SELECT id, codigo_catastral, propietario, estado_gestion, area_ha 
                FROM predios_gestion 
                WHERE propietario ILIKE :s OR codigo_catastral ILIKE :s
                LIMIT 10
            """)
            with engine.connect() as conn:
                results = pd.read_sql(query_search, conn, params={"s": f"%{search_term}%"})
            
            if not results.empty:
                # Selector de resultados
                predio_opt = {f"{r['propietario']} - {r['codigo_catastral']} (ID:{r['id']})": r['id'] for _, r in results.iterrows()}
                selected_id_key = st.selectbox("Seleccione el predio encontrado:", list(predio_opt.keys()))
                
                if selected_id_key:
                    selected_id = predio_opt[selected_id_key]
                    # Datos actuales
                    row = results[results['id'] == selected_id].iloc[0]
                    
                    st.info(f"Gestionando: **{row['propietario']}**")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Área (ha)", f"{row['area_ha']:.2f}")
                    c2.metric("Estado Actual", row['estado_gestion'])
                    c3.metric("Código", row['codigo_catastral'])
                    
                    # Formulario de actualización
                    with st.form("update_predio"):
                        new_status = st.selectbox(
                            "Nuevo Estado:", 
                            ["Identificado", "En Negociación", "Viabilidad Técnica", "Ejecutado / Conservado", "Descartado"],
                            index=0
                        )
                        observacion = st.text_input("Notas de gestión (Opcional):")
                        
                        if st.form_submit_button("💾 Actualizar Estado"):
                            try:
                                sql_upd = text("UPDATE predios_gestion SET estado_gestion = :st WHERE id = :id")
                                with engine.connect() as conn:
                                    conn.execute(sql_upd, {"st": new_status, "id": selected_id})
                                    conn.commit()
                                st.success(f"✅ Predio actualizado a: {new_status}")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error al actualizar: {e}")
            else:
                st.warning("No se encontraron predios con ese criterio.")
                
    except Exception as e:
        st.error(f"Error en módulo de gestión: {e}")

    st.divider()

    # SECCIÓN B: SINCRONIZACIÓN (MANTENIDA)
    with st.expander("☁️ Sincronización Masiva (GitHub -> DB)", expanded=False):
        st.markdown("Use esto solo para recargar toda la base de datos desde el archivo original.")
        GITHUB_URL = "https://raw.githubusercontent.com/omejiariv/SIHCLI-POTER-2026/main/data/PrediosEjecutados.geojson"
        
        if st.button("🔄 Forzar Sincronización Completa"):
            with st.status("Procesando...", expanded=True) as status:
                try:
                    engine = get_engine()
                    gdf = gpd.read_file(GITHUB_URL)
                    if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                    
                    count = 0
                    with engine.connect() as conn:
                        conn.execute(text("TRUNCATE TABLE predios_gestion RESTART IDENTITY"))
                        for idx, row in gdf.iterrows():
                            cod = row.get('codigo_catastral') or row.get('codigo') or str(idx)
                            prop = row.get('propietario') or 'No Registrado'
                            est = row.get('estado') or 'Identificado'
                            area = row.get('area_ha') or row.get('area') or 0.0
                            geom_wkt = row.geometry.wkt
                            
                            sql_ins = text("""
                                INSERT INTO predios_gestion (codigo_catastral, propietario, estado_gestion, area_ha, geom)
                                VALUES (:c, :p, :e, :a, ST_Multi(ST_GeomFromText(:g, 4326)))
                            """)
                            conn.execute(sql_ins, {"c": cod, "p": prop, "e": est, "a": area, "g": geom_wkt})
                            count += 1
                        conn.commit()
                    status.update(label="¡Completado!", state="complete")
                    st.success(f"✅ Base de datos reiniciada con {count} predios.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==============================================================================
# TAB 3: CONSOLA SQL
# ==============================================================================
with tab_sql:
    st.warning("Consola de administración directa.")
    query = st.text_area("SQL:", height=150)
    if st.button("▶️ Ejecutar"):
        if query:
            try:
                engine = get_engine()
                with engine.connect() as conn:
                    if query.strip().lower().startswith("select"):
                        st.dataframe(pd.read_sql(text(query), conn))
                    else:
                        res = conn.execute(text(query))
                        conn.commit()
                        st.success(f"✅ Filas afectadas: {res.rowcount}")
            except Exception as e:
                st.error(f"Error: {e}")