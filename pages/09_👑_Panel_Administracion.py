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
    # Conecta directamente a Supabase usando el secreto configurado
    return create_engine(st.secrets["DATABASE_URL"])

# --- 3. INTERFAZ PRINCIPAL ---
st.title("👑 Panel de Administración y Edición de Datos")
st.markdown("---")

tab_est, tab_predios, tab_sql = st.tabs([
    "📡 Estaciones (Editor)", 
    "🏡 Predios (Sincronización Nube)", 
    "🛠️ Consola SQL"
])

# ==============================================================================
# TAB 1: EDITOR DE ESTACIONES (Blindado para tus columnas)
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
                # Normalización de nombres de columnas para evitar errores de mayúsculas/minúsculas
                col_map = {c.lower().strip(): c for c in df_est.columns}
                
                # Mapeo inteligente usando tus columnas reales
                col_lat = col_map.get('latitude') or col_map.get('latitud')
                col_lon = col_map.get('longitude') or col_map.get('longitud')
                col_nom = col_map.get('nom_est')
                col_mun = col_map.get('municipio')
                
                curr = df_est.iloc[0]
                
                with st.form("form_estacion"):
                    c1, c2 = st.columns(2)
                    
                    val_nom = curr[col_nom] if col_nom else ""
                    val_mun = curr[col_mun] if col_mun else ""
                    # Conversión segura a float
                    val_lat = float(curr[col_lat]) if col_lat and pd.notnull(curr[col_lat]) else 0.0
                    val_lon = float(curr[col_lon]) if col_lon and pd.notnull(curr[col_lon]) else 0.0
                    
                    new_name = c1.text_input("Nombre:", value=val_nom)
                    new_muni = c2.text_input("Municipio:", value=val_mun)
                    new_lat = c1.number_input(f"Latitud ({col_lat}):", value=val_lat, format="%.6f")
                    new_lon = c2.number_input(f"Longitud ({col_lon}):", value=val_lon, format="%.6f")
                    
                    st.caption("ℹ️ Al guardar, se actualizarán las coordenadas y la geometría espacial en Supabase.")
                    
                    if st.form_submit_button("💾 Guardar Cambios"):
                        if col_lat and col_lon:
                            try:
                                # Actualiza columnas planas y reconstruye el objeto geométrico (ST_Point)
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
                                st.success("✅ Estación actualizada exitosamente en la nube.")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error SQL: {e}")
                        else:
                            st.error("No se encontraron las columnas de coordenadas en la tabla.")

    except Exception as e:
        st.error(f"Error de conexión: {e}")

# ==============================================================================
# TAB 2: GESTIÓN DE PREDIOS (GitHub -> Supabase)
# ==============================================================================
with tab_predios:
    st.subheader("🏡 Sincronización de Predios")
    st.markdown("Importa los predios gestionados directamente desde tu repositorio oficial.")
    
    # URL RAW de GitHub (La versión 'cruda' del archivo JSON)
    GITHUB_URL = "https://raw.githubusercontent.com/omejiariv/SIHCLI-POTER-2026/main/data/PrediosEjecutados.geojson"
    
    st.info(f"🔗 Fuente Conectada: **GitHub / SIHCLI-POTER-2026**")
    
    # Verificar si la tabla existe
    try:
        engine = get_engine()
        with engine.connect() as conn:
            exists = conn.execute(text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'predios_gestion')")).scalar()
        
        if not exists:
            st.warning("⚠️ La tabla 'predios_gestion' no existe en Supabase.")
            st.markdown("Por favor, ejecuta el script de creación en la pestaña **🛠️ Consola SQL**.")
        else:
            col_a, col_b = st.columns([1, 2])
            with col_a:
                if st.button("☁️ Sincronizar desde GitHub", type="primary"):
                    with st.status("Procesando datos...", expanded=True) as status:
                        try:
                            # 1. Leer directamente desde la URL de GitHub
                            st.write("📥 Descargando GeoJSON...")
                            gdf = gpd.read_file(GITHUB_URL)
                            
                            # 2. Ajustar proyección
                            if gdf.crs != "EPSG:4326":
                                gdf = gdf.to_crs("EPSG:4326")
                            st.write(f"✅ Archivo leído: {len(gdf)} predios encontrados.")
                            
                            # 3. Insertar en Supabase
                            st.write("💾 Escribiendo en base de datos...")
                            count = 0
                            with engine.connect() as conn:
                                # Opcional: Limpiar tabla antes de cargar (descomentar si quieres reemplazar todo)
                                # conn.execute(text("TRUNCATE TABLE predios_gestion")) 
                                
                                for idx, row in gdf.iterrows():
                                    # Mapeo de campos flexible (ajusta según tu GeoJSON real)
                                    cod = row.get('codigo_catastral') or row.get('codigo') or str(idx)
                                    prop = row.get('propietario') or 'No Registrado'
                                    est = row.get('estado') or 'Identificado'
                                    area = row.get('area_ha') or row.get('area') or 0.0
                                    geom_wkt = row.geometry.wkt
                                    
                                    sql_ins = text("""
                                        INSERT INTO predios_gestion (codigo_catastral, propietario, estado_gestion, area_ha, geom)
                                        VALUES (:c, :p, :e, :a, ST_GeomFromText(:g, 4326))
                                        ON CONFLICT DO NOTHING -- Evita duplicados si ya existen
                                    """)
                                    conn.execute(sql_ins, {"c": cod, "p": prop, "e": est, "a": area, "g": geom_wkt})
                                    count += 1
                                conn.commit()
                            
                            status.update(label="¡Sincronización Completada!", state="complete", expanded=False)
                            st.success(f"✅ Se procesaron {count} registros correctamente.")
                            
                        except Exception as e:
                            st.error(f"Error en la sincronización: {e}")

    except Exception as e:
        st.error(f"Error verificando base de datos: {e}")

# ==============================================================================
# TAB 3: CONSOLA SQL
# ==============================================================================
with tab_sql:
    st.warning("Consola de administración directa de base de datos.")
    
    # Pre-carga del script de creación de tabla por si se necesita
    default_sql = """CREATE TABLE IF NOT EXISTS predios_gestion (
    id SERIAL PRIMARY KEY,
    codigo_catastral VARCHAR(50),
    propietario VARCHAR(100),
    estado_gestion VARCHAR(50), 
    area_ha FLOAT,
    geom GEOMETRY(Polygon, 4326)
);"""
    
    query = st.text_area("SQL:", value=default_sql, height=150)
    
    if st.button("▶️ Ejecutar SQL"):
        if query:
            try:
                engine = get_engine()
                with engine.connect() as conn:
                    if query.strip().lower().startswith("select"):
                        st.dataframe(pd.read_sql(text(query), conn))
                    else:
                        res = conn.execute(text(query))
                        conn.commit()
                        st.success(f"✅ Comando ejecutado exitosamente.")
            except Exception as e:
                st.error(f"Error SQL: {e}")