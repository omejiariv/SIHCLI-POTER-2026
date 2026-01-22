# pages/09_👑_Panel_Administracion.py

import streamlit as st
import pandas as pd
import json
import io
import time
import sys
import os
import tempfile
import zipfile
import geopandas as gpd
from sqlalchemy import text
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape

# --- CONFIGURACIÓN DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from modules.db_manager import get_engine
except ImportError:
    from db_manager import get_engine

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Panel de Administración", page_icon="👑", layout="wide")

# --- AUTENTICACIÓN (ORIGINAL) ---
def check_password():
    """Valida usuario/contraseña contra secrets.toml"""
    if st.session_state.get("password_correct", False):
        return True
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Acceso Restringido")
        st.info("Panel de Control SIHCLI-POTER (Nube)")
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
                st.error("🚫 Acceso Denegado")
                return False
    return False

if not check_password():
    st.stop()

# --- NUEVA FUNCIÓN: CARGA GIS ROBUSTA (CORRIGE COORDENADAS) ---
def cargar_capa_gis_robusta(uploaded_file, nombre_tabla, engine):
    if uploaded_file is None: return
    
    status = st.status(f"🚀 Procesando {nombre_tabla}...", expanded=True)
    try:
        # 1. Guardar temporalmente
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # 2. Leer archivo
        gdf = None
        if suffix == '.zip':
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                for root, dirs, files in os.walk(tmp_dir):
                    for file in files:
                        if file.endswith(".shp"):
                            gdf = gpd.read_file(os.path.join(root, file))
                            break
        else:
            gdf = gpd.read_file(tmp_path)
            
        if gdf is None:
            status.error("No se pudo leer el archivo geográfico.")
            return

        status.write(f"✅ Leído: {len(gdf)} registros. CRS original: {gdf.crs}")

        # 3. REPROYECCIÓN OBLIGATORIA A WGS84 (Lat/Lon)
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            status.write("🔄 Reproyectando a WGS84 (EPSG:4326)...")
            gdf = gdf.to_crs("EPSG:4326")
        
        # 4. Normalización de columnas
        gdf.columns = [c.lower() for c in gdf.columns]
        
        # Mapeo inteligente de columnas
        rename_map = {}
        if 'bocatomas' in nombre_tabla:
            if 'nombre' in gdf.columns: rename_map['nombre'] = 'nom_bocatoma'
        elif 'suelos' in nombre_tabla:
            if 'gridcode' in gdf.columns: rename_map['gridcode'] = 'codigo'
            if 'simbolo' in gdf.columns: rename_map['simbolo'] = 'codigo'
        elif 'zonas_hidrogeologicas' in nombre_tabla:
            if 'nombre' in gdf.columns: rename_map['nombre'] = 'nombre_zona'
            
        if rename_map:
            gdf = gdf.rename(columns=rename_map)

        # 5. Carga a PostGIS
        status.write("📤 Subiendo a Base de Datos...")
        gdf.to_postgis(nombre_tabla, engine, if_exists='replace', index=False)
        
        status.update(label="¡Carga Exitosa!", state="complete", expanded=False)
        st.success(f"Capa **{nombre_tabla}** actualizada ({len(gdf)} registros).")
        if len(gdf) > 0: st.balloons()
        
    except Exception as e:
        status.update(label="Error", state="error")
        st.error(f"Error crítico: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# --- INTERFAZ PRINCIPAL ---
st.title("👑 Panel de Administración y Edición de Datos")
st.markdown("---")

engine = get_engine()

# DEFINICIÓN DE PESTAÑAS (INCLUYENDO LA NUEVA DE BOCATOMAS)
tab_est, tab_indices, tab_predios, tab_cuencas, tab_mun, tab_boca, tab_hidro, tab_suelos, tab_sql = st.tabs([
    "📡 Estaciones", "📊 Índices", "🏠 Predios", "🌊 Cuencas", 
    "🏙️ Municipios", "💧 Bocatomas", "⛰️ Hidrogeología", "🌱 Suelos", "🛠️ SQL"
])

# ==============================================================================
# TAB 1: ESTACIONES (TU LÓGICA ORIGINAL RESTAURADA)
# ==============================================================================
with tab_est:
    st.header("Gestión de Estaciones Hidroclimáticas")
    sub_editar, sub_crear, sub_carga = st.tabs(["✏️ Editar Existente", "➕ Crear Nueva", "📂 Carga Masiva"])
    
    # --- Sub-pestaña Editar ---
    with sub_editar:
        st.info("Busca una estación para corregir sus coordenadas.")
        if engine:
            with engine.connect() as conn:
                df_l = pd.read_sql(text("SELECT id_estacion, nom_est FROM estaciones ORDER BY nom_est"), conn)
                if not df_l.empty:
                    df_l['display'] = df_l['nom_est'] + " (" + df_l['id_estacion'].astype(str) + ")"
                    seleccion = st.selectbox("Buscar Estación:", df_l['display'].tolist(), index=None)
                    
                    if seleccion:
                        id_sel = seleccion.split('(')[-1].replace(')', '').strip()
                        df_f = pd.read_sql(text("SELECT * FROM estaciones WHERE id_estacion = :id"), conn, params={"id": id_sel})
                        df_f.columns = [c.lower() for c in df_f.columns]
                        
                        if not df_f.empty:
                            est_data = df_f.iloc[0]
                            st.divider()
                            st.subheader(f"Editando: {est_data.get('nom_est')}")
                            
                            with st.form("edit_est"):
                                c1, c2 = st.columns(2)
                                with c1:
                                    nn = st.text_input("Nombre", value=est_data.get('nom_est', ''))
                                    
                                    cat_actual = est_data.get('categoria', 'Pluviométrica')
                                    opt_cat = ["Pluviométrica", "Limnimétrica", "Climática", "Otras"]
                                    idx_cat = opt_cat.index(cat_actual) if cat_actual in opt_cat else 0
                                    nc = st.selectbox("Categoría", opt_cat, index=idx_cat)
                                    
                                    tec_actual = est_data.get('tecnologia', 'Convencional')
                                    opt_tec = ["Convencional", "Automática", "Radar"]
                                    idx_tec = opt_tec.index(tec_actual) if tec_actual in opt_tec else 0
                                    nt = st.selectbox("Tecnología", opt_tec, index=idx_tec)
                                    
                                    nm = st.text_input("Municipio", value=est_data.get('municipio', ''))

                                with c2:
                                    nl = st.number_input("Latitud", value=float(est_data.get('latitud') or 0.0), format="%.5f")
                                    nlo = st.number_input("Longitud", value=float(est_data.get('longitud') or 0.0), format="%.5f")
                                    ne = st.number_input("Elevación", value=float(est_data.get('elevacion') or 0.0))
                                    st.text_input("ID", value=est_data.get('id_estacion'), disabled=True)

                                if st.form_submit_button("Guardar Cambios"):
                                    conn.execute(text("""
                                        UPDATE estaciones 
                                        SET nom_est=:n, categoria=:c, tecnologia=:t, municipio=:m, 
                                            latitud=:la, longitud=:lo, elevacion=:e 
                                        WHERE id_estacion=:id
                                    """), {"n": nn, "c": nc, "t": nt, "m": nm, "la": nl, "lo": nlo, "e": ne, "id": id_sel})
                                    conn.commit()
                                    st.success("Estación actualizada.")
                                    time.sleep(1)
                                    st.rerun()

    # --- Sub-pestaña Crear ---
    with sub_crear:
        st.markdown("### Registrar Nueva Estación")
        with st.form("new_est"):
            c1, c2 = st.columns(2)
            nid = c1.text_input("ID (Único)")
            nnom = c1.text_input("Nombre")
            nmun = c1.text_input("Municipio")
            nlat = c2.number_input("Latitud", value=6.0, format="%.5f")
            nlon = c2.number_input("Longitud", value=-75.0, format="%.5f")
            nelev = c2.number_input("Elevación", value=1500.0)
            
            if st.form_submit_button("Crear Estación"):
                if nid and nnom:
                    with engine.connect() as conn:
                        try:
                            conn.execute(text("""
                                INSERT INTO estaciones (id_estacion, nom_est, municipio, latitud, longitud, elevacion)
                                VALUES (:id, :n, :m, :la, :lo, :e)
                            """), {"id": nid, "n": nnom, "m": nmun, "la": nlat, "lo": nlon, "e": nelev})
                            conn.commit()
                            st.success("Estación creada exitosamente.")
                        except Exception as e: st.error(f"Error: {e}")

    # --- Sub-pestaña Carga ---
    with sub_carga:
        st.markdown("### Carga de Metadatos")
        st.info("Sube 'mapaCVENSO.csv' para actualizar metadatos.")
        up_meta = st.file_uploader("CSV Metadatos", type=["csv"])
        if up_meta and st.button("Procesar Metadatos"):
            try:
                df = pd.read_csv(up_meta, sep=';', encoding='latin-1', engine='python')
                df.columns = [c.strip() for c in df.columns]
                
                with engine.connect() as conn:
                    count = 0
                    for _, row in df.iterrows():
                        try:
                            sid = str(row['Id_estacio']).strip()
                            snom = str(row['Nom_Est']).strip()
                            slat = float(str(row['Latitud_geo']).replace(',', '.'))
                            slon = float(str(row['Longitud_geo']).replace(',', '.'))
                            conn.execute(text("""
                                INSERT INTO estaciones (id_estacion, nom_est, latitud, longitud)
                                VALUES (:id, :n, :la, :lo)
                                ON CONFLICT (id_estacion) DO UPDATE SET
                                nom_est = EXCLUDED.nom_est, latitud = EXCLUDED.latitud, longitud = EXCLUDED.longitud
                            """), {"id": sid, "n": snom, "la": slat, "lo": slon})
                            count += 1
                        except: pass
                    conn.commit()
                st.success(f"Procesados {count} registros.")
            except Exception as e: st.error(f"Error: {e}")

# ==============================================================================
# TAB 2: ÍNDICES (TU LÓGICA ORIGINAL RESTAURADA)
# ==============================================================================
with tab_indices:
    st.header("📊 Gestión de Índices Climáticos")
    up_idx = st.file_uploader("CSV Índices (ONI, SOI, IOD)", type=["csv"])
    if up_idx and st.button("Cargar Índices"):
        try:
            df = pd.read_csv(up_idx, engine='python')
            df.columns = [c.lower().strip() for c in df.columns]
            if 'id' in df.columns: df = df.drop(columns=['id'])
            df.to_sql('indices_climaticos', engine, if_exists='replace', index=False)
            st.success(f"Cargados {len(df)} registros.")
        except Exception as e: st.error(str(e))

# ==============================================================================
# TAB 3: PREDIOS (TU LÓGICA ORIGINAL RESTAURADA)
# ==============================================================================
with tab_predios:
    st.header("🏠 Gestión de Predios")
    sb1, sb2, sb3 = st.tabs(["Editar", "Crear", "Carga GeoJSON"])
    
    with sb1: # Editar
        with engine.connect() as conn:
            df_l = pd.read_sql(text("SELECT id_predio, nombre_predio FROM predios ORDER BY nombre_predio"), conn)
            if not df_l.empty:
                df_l['d'] = df_l['nombre_predio'] + " (" + df_l['id_predio'].astype(str) + ")"
                sel = st.selectbox("Buscar Predio:", df_l['d'].tolist(), index=None)
                if sel:
                    id_p = sel.split('(')[-1].replace(')', '')
                    d = pd.read_sql(text("SELECT * FROM predios WHERE id_predio=:id"), conn, params={"id": id_p}).iloc[0]
                    with st.form("edit_pred"):
                        nn = st.text_input("Nombre", value=d['nombre_predio'])
                        nm = st.text_input("Municipio", value=d.get('municipio', ''))
                        na = st.number_input("Área (ha)", value=float(d.get('area_ha') or 0))
                        if st.form_submit_button("Actualizar"):
                            conn.execute(text("UPDATE predios SET nombre_predio=:n, municipio=:m, area_ha=:a WHERE id_predio=:id"),
                                        {"n": nn, "m": nm, "a": na, "id": id_p})
                            conn.commit()
                            st.success("Predio actualizado.")
                            st.rerun()

    with sb3: # Carga
        up_gp = st.file_uploader("GeoJSON Predios", type=["geojson", "json"])
        if up_gp and st.button("Procesar Predios"):
            try:
                data = json.load(up_gp)
                rows = []
                for f in data['features']:
                    p = f.get('properties', {})
                    rows.append({
                        "id_predio": str(p.get('PK_PREDIOS', 'SN')),
                        "nombre_predio": p.get('NOMBRE_PRE', 'Sin Nombre'),
                        "municipio": p.get('NOMB_MPIO', ''),
                        "area_ha": float(p.get('AREA_HA', 0))
                    })
                pd.DataFrame(rows).drop_duplicates('id_predio').to_sql('predios', engine, if_exists='append', index=False, method='multi')
                st.success("Predios cargados.")
            except Exception as e: st.error(f"Error: {e}")

# ==============================================================================
# TAB 4: CUENCAS (TU LÓGICA ORIGINAL RESTAURADA)
# ==============================================================================
with tab_cuencas:
    st.header("🌊 Gestión de Cuencas")
    sb1, sb2 = st.tabs(["Editar", "Carga GeoJSON"])
    
    with sb2:
        up_c = st.file_uploader("GeoJSON Cuencas", type=["geojson", "json"])
        if up_c and st.button("Procesar Cuencas"):
            try:
                data = json.load(up_c)
                rows = []
                for f in data['features']:
                    p = f.get('properties', {})
                    area = float(p.get('Shape_Area', 0)) / 1_000_000
                    rows.append({
                        "id_cuenca": str(p.get('COD', 'SN')),
                        "nombre_cuenca": p.get('SUBC_LBL', 'Sin Nombre'),
                        "area_km2": area,
                        "rio_principal": p.get('SZH', '')
                    })
                pd.DataFrame(rows).drop_duplicates('id_cuenca').to_sql('cuencas', engine, if_exists='append', index=False, method='multi')
                st.success("Cuencas cargadas.")
            except Exception as e: st.error(f"Error: {e}")

# ==============================================================================
# TAB 5: MUNICIPIOS (TU LÓGICA ORIGINAL RESTAURADA)
# ==============================================================================
with tab_mun:
    st.header("🏙️ Gestión de Municipios")
    sb1, sb2 = st.tabs(["Editar", "Carga GeoJSON"])
    
    with sb2:
        up_m = st.file_uploader("GeoJSON Municipios", type=["geojson", "json"])
        if up_m and st.button("Procesar Municipios"):
            try:
                data = json.load(up_m)
                rows = []
                for f in data['features']:
                    p = f.get('properties', {})
                    rows.append({
                        "id_municipio": str(p.get('MPIO_CDPMP', '00')),
                        "nombre_municipio": p.get('MPIO_CNMBR', ''),
                        "departamento": p.get('DPTO_CNMBR', 'Antioquia')
                    })
                pd.DataFrame(rows).drop_duplicates('id_municipio').to_sql('municipios', engine, if_exists='append', index=False, method='multi')
                st.success("Municipios cargados.")
            except Exception as e: st.error(f"Error: {e}")

# ==============================================================================
# TAB 6: BOCATOMAS (¡NUEVO Y ROBUSTO!)
# ==============================================================================
with tab_boca:
    st.header("💧 Gestión de Bocatomas")
    st.info("Sube tu archivo 'Bocatomas_Ant.shp' (en ZIP) o GeoJSON. El sistema corregirá las coordenadas.")
    f_boca = st.file_uploader("Archivo Bocatomas", type=["zip", "geojson", "kml"])
    
    if st.button("Cargar Bocatomas"):
        cargar_capa_gis_robusta(f_boca, "bocatomas", engine)
        
    st.divider()
    try:
        c = pd.read_sql("SELECT count(*) FROM bocatomas", engine).iloc[0,0]
        st.metric("Registros en BD", c)
    except: st.warning("La tabla aún no existe.")

# ==============================================================================
# TAB 7: HIDROGEOLOGÍA (MEJORADO CON FUNCIÓN ROBUSTA)
# ==============================================================================
with tab_hidro:
    st.header("⛰️ Gestión Hidrogeológica")
    st.info("Sube 'Zonas_PotHidrogeologico.geojson'.")
    f_hidro = st.file_uploader("Archivo Zonas Hidro", type=["geojson", "zip"])
    
    if st.button("Cargar Hidrogeología"):
        cargar_capa_gis_robusta(f_hidro, "zonas_hidrogeologicas", engine)

# ==============================================================================
# TAB 8: SUELOS (MEJORADO CON FUNCIÓN ROBUSTA)
# ==============================================================================
with tab_suelos:
    st.header("🌱 Gestión de Suelos")
    st.info("Sube 'Suelos_Antioquia.geojson'.")
    f_suelo = st.file_uploader("Archivo Suelos", type=["geojson", "zip"])
    
    if st.button("Cargar Suelos"):
        cargar_capa_gis_robusta(f_suelo, "suelos", engine)

# ==============================================================================
# TAB 9: SQL
# ==============================================================================
with tab_sql:
    st.header("🛠️ Consola SQL")
    q = st.text_area("Query:")
    if st.button("Ejecutar"):
        try:
            with engine.connect() as conn:
                if q.strip().lower().startswith("select"):
                    st.dataframe(pd.read_sql(text(q), conn))
                else:
                    conn.execute(text(q))
                    conn.commit()
                    st.success("Hecho.")
        except Exception as e: st.error(str(e))