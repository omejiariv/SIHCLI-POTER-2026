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
import rasterio
from sqlalchemy import text
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape
import shutil

# --- 1. CONFIGURACIÓN DE RUTAS E IMPORTACIONES ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from modules.db_manager import get_engine
except ImportError:
    from db_manager import get_engine

st.set_page_config(page_title="Panel de Administración", page_icon="👑", layout="wide")

# --- 2. AUTENTICACIÓN ---
def check_password():
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

engine = get_engine()

# --- 3. FUNCIONES AUXILIARES ROBUSTAS ---

def cargar_capa_gis_robusta(uploaded_file, nombre_tabla, engine):
    """Carga archivos GIS, repara coordenadas y sube a BD."""
    if uploaded_file is None: return
    
    status = st.status(f"🚀 Procesando {nombre_tabla}...", expanded=True)
    try:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
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

        # REPROYECCIÓN OBLIGATORIA A WGS84
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            status.write("🔄 Reproyectando a WGS84 (EPSG:4326)...")
            gdf = gdf.to_crs("EPSG:4326")
        
        # Normalización
        gdf.columns = [c.lower() for c in gdf.columns]
        
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

# --- 4. INTERFAZ PRINCIPAL ---
st.title("👑 Panel de Administración y Edición de Datos")
st.markdown("---")

# DEFINICIÓN DE PESTAÑAS (TODAS INCLUIDAS)
tabs = st.tabs([
    "📡 Estaciones", "📊 Índices", "🏠 Predios", "🌊 Cuencas", 
    "🏙️ Municipios", "🌲 Coberturas", "💧 Bocatomas", "⛰️ Hidrogeología", "🌱 Suelos", "🛠️ SQL"
])

# ==============================================================================
# TAB 1: ESTACIONES (LÓGICA ORIGINAL COMPLETA)
# ==============================================================================
with tabs[0]:
    st.header("📡 Gestión de Estaciones")
    sub_editar, sub_crear, sub_carga = st.tabs(["✏️ Editar Existente", "➕ Crear Nueva", "📂 Carga Masiva"])
    
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
                        
                        if not df_f.empty:
                            est_data = df_f.iloc[0]
                            with st.form("edit_est"):
                                c1, c2 = st.columns(2)
                                with c1:
                                    nn = st.text_input("Nombre", value=est_data.get('nom_est', ''))
                                    nm = st.text_input("Municipio", value=est_data.get('municipio', ''))
                                    
                                    cat_actual = est_data.get('categoria', 'Pluviométrica')
                                    opt_cat = ["Pluviométrica", "Limnimétrica", "Climática", "Otras"]
                                    idx_cat = opt_cat.index(cat_actual) if cat_actual in opt_cat else 0
                                    nc = st.selectbox("Categoría", opt_cat, index=idx_cat)
                                    
                                    tec_actual = est_data.get('tecnologia', 'Convencional')
                                    opt_tec = ["Convencional", "Automática", "Radar"]
                                    idx_tec = opt_tec.index(tec_actual) if tec_actual in opt_tec else 0
                                    nt = st.selectbox("Tecnología", opt_tec, index=idx_tec)

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
                                    time.sleep(0.5)
                                    st.rerun()

    with sub_crear:
        with st.form("new_est"):
            c1, c2 = st.columns(2)
            nid = c1.text_input("ID (Único)")
            nnom = c1.text_input("Nombre")
            nlat = c2.number_input("Latitud", value=6.0, format="%.5f")
            nlon = c2.number_input("Longitud", value=-75.0, format="%.5f")
            
            if st.form_submit_button("Crear Estación"):
                if nid and nnom:
                    with engine.connect() as conn:
                        try:
                            conn.execute(text("INSERT INTO estaciones (id_estacion, nom_est, latitud, longitud) VALUES (:id, :n, :la, :lo)"),
                                        {"id": nid, "n": nnom, "la": nlat, "lo": nlon})
                            conn.commit()
                            st.success("Creada.")
                        except Exception as e: st.error(f"Error: {e}")

    with sub_carga:
        st.info("Carga 'mapaCVENSO.csv' (Actualiza coordenadas).")
        up_meta = st.file_uploader("CSV Metadatos", type=["csv"])
        if up_meta and st.button("Procesar"):
            try:
                df = pd.read_csv(up_meta, sep=';', encoding='latin-1')
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
# TAB 2: ÍNDICES (MEJORADO CON EDITOR)
# ==============================================================================
with tabs[1]:
    st.header("📊 Índices Climáticos")
    sb1, sb2 = st.tabs(["👁️ Ver/Editar", "📂 Cargar CSV"])
    
    with sb1:
        try:
            df_idx = pd.read_sql("SELECT * FROM indices_climaticos ORDER BY fecha DESC LIMIT 1000", engine)
            st.info("Edita los valores directamente en la tabla.")
            df_editado = st.data_editor(df_idx, num_rows="dynamic", key="editor_indices")
            if st.button("💾 Guardar Cambios Índices"):
                df_editado.to_sql('indices_climaticos', engine, if_exists='replace', index=False)
                st.success("Actualizado.")
        except: st.warning("Sin datos.")

    with sb2:
        up_idx = st.file_uploader("CSV Índices", type=["csv"])
        if up_idx and st.button("Cargar"):
            df = pd.read_csv(up_idx)
            df.columns = [c.lower().strip() for c in df.columns]
            if 'id' in df.columns: df = df.drop(columns=['id'])
            df.to_sql('indices_climaticos', engine, if_exists='replace', index=False)
            st.success("Cargado.")

# ==============================================================================
# TAB 3: PREDIOS (MEJORADO CON EDITOR)
# ==============================================================================
with tabs[2]:
    st.header("🏠 Gestión de Predios")
    sb1, sb2, sb3 = st.tabs(["👁️ Tabla Editable", "➕ Crear (Formulario)", "📂 Carga GeoJSON"])
    
    with sb1:
        try:
            df_p = pd.read_sql("SELECT * FROM predios LIMIT 2000", engine)
            st.data_editor(df_p, key="ed_predios")
            st.caption("Nota: Edición masiva habilitada solo vía carga GeoJSON por seguridad.")
        except: st.warning("Sin datos.")

    with sb2:
        with st.form("new_pred"):
            c1, c2 = st.columns(2)
            pid = c1.text_input("ID Predio")
            pnom = c1.text_input("Nombre")
            pmun = c2.text_input("Municipio")
            parea = c2.number_input("Área (ha)")
            if st.form_submit_button("Guardar"):
                with engine.connect() as conn:
                    conn.execute(text("INSERT INTO predios (id_predio, nombre_predio, municipio, area_ha) VALUES (:id, :n, :m, :a)"),
                                {"id": pid, "n": pnom, "m": pmun, "a": parea})
                    conn.commit()
                st.success("Guardado.")

    with sb3:
        up_gp = st.file_uploader("GeoJSON Predios", type=["geojson", "json"])
        if up_gp and st.button("Procesar Predios"):
            try:
                data = json.load(up_gp)
                rows = []
                for f in data['features']:
                    p = f.get('properties', {})
                    rows.append({
                        "id_predio": str(p.get('PK_PREDIOS', 'SN')),
                        "nombre_predio": p.get('NOMBRE_PRE', ''),
                        "municipio": p.get('NOMB_MPIO', ''),
                        "area_ha": float(p.get('AREA_HA', 0))
                    })
                # Usamos INSERT simple para evitar complejidad, idealmente UPSERT
                pd.DataFrame(rows).drop_duplicates('id_predio').to_sql('predios', engine, if_exists='append', index=False, method='multi')
                st.success("Cargado.")
            except: st.warning("Error de carga o duplicados. Use SQL para limpieza avanzada.")

# ==============================================================================
# TAB 4: CUENCAS (MEJORADO: EDITOR + FIX DUPLICADOS)
# ==============================================================================
with tabs[3]:
    st.header("🌊 Gestión de Cuencas")
    sb1, sb2, sb3 = st.tabs(["👁️ Tabla Editable", "➕ Crear", "📂 Carga GeoJSON"])
    
    with sb1:
        try:
            df_c = pd.read_sql("SELECT * FROM cuencas", engine)
            st.data_editor(df_c, key="ed_cuencas")
        except: st.write("Sin datos.")

    with sb3:
        st.info("Carga 'SubcuencasAinfluencia.geojson'. Corrige duplicados automáticamente.")
        up_c = st.file_uploader("GeoJSON Cuencas", type=["geojson", "json"])
        if up_c and st.button("Procesar Cuencas"):
            try:
                data = json.load(up_c)
                count = 0
                with engine.connect() as conn:
                    for f in data['features']:
                        p = f.get('properties', {})
                        area = float(p.get('Shape_Area', 0)) / 1_000_000
                        row = {
                            "id": str(p.get('COD', 'SN')),
                            "nom": p.get('SUBC_LBL', 'Sin Nombre'),
                            "area": area,
                            "rio": p.get('SZH', '')
                        }
                        # FIX DUPLICADOS: UPSERT
                        conn.execute(text("""
                            INSERT INTO cuencas (id_cuenca, nombre_cuenca, area_km2, rio_principal)
                            VALUES (:id, :nom, :area, :rio)
                            ON CONFLICT (id_cuenca) DO UPDATE SET 
                            nombre_cuenca = EXCLUDED.nombre_cuenca, area_km2 = EXCLUDED.area_km2
                        """), row)
                        count += 1
                    conn.commit()
                st.success(f"Procesados {count} registros.")
            except Exception as e: st.error(f"Error: {e}")

# ==============================================================================
# TAB 5: MUNICIPIOS (LÓGICA ORIGINAL)
# ==============================================================================
with tabs[4]:
    st.header("🏙️ Municipios")
    up_m = st.file_uploader("GeoJSON Municipios", type=["geojson", "json"])
    if up_m and st.button("Cargar"):
        st.info("Carga disponible.")

# ==============================================================================
# TAB 6: COBERTURAS (¡NUEVO!)
# ==============================================================================
with tabs[5]:
    st.header("🌲 Coberturas Vegetales")
    f_tiff = st.file_uploader("Cargar 'Cob25m_WGS84.tiff'", type=["tiff", "tif"])
    if f_tiff and st.button("Guardar Coberturas"):
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", "coberturas.tif")
        with open(path, "wb") as f:
            f.write(f_tiff.getbuffer())
        st.success("Archivo guardado. Disponible para el mapa.")

# ==============================================================================
# TABS 7, 8, 9: GIS ROBUSTO
# ==============================================================================
with tabs[6]: # Bocatomas
    st.header("💧 Bocatomas")
    f_boca = st.file_uploader("Archivo Bocatomas (ZIP/GeoJSON)", type=["zip", "geojson"])
    if st.button("Cargar Bocatomas"): cargar_capa_gis_robusta(f_boca, "bocatomas", engine)

with tabs[7]: # Hidro
    st.header("⛰️ Hidrogeología")
    f_hidro = st.file_uploader("Archivo Hidro (GeoJSON)", type=["geojson", "zip"])
    if st.button("Cargar Hidro"): cargar_capa_gis_robusta(f_hidro, "zonas_hidrogeologicas", engine)

with tabs[8]: # Suelos
    st.header("🌱 Suelos")
    f_suelo = st.file_uploader("Archivo Suelos (GeoJSON)", type=["geojson", "zip"])
    if st.button("Cargar Suelos"): cargar_capa_gis_robusta(f_suelo, "suelos", engine)

# ==============================================================================
# TAB 10: SQL
# ==============================================================================
with tabs[9]:
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