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

from modules.admin_utils import get_raster_list, upload_raster_to_storage, delete_raster_from_storage
from supabase import create_client


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

# --- 3. FUNCIONES AUXILIARES ---

def cargar_capa_gis_robusta(uploaded_file, nombre_tabla, engine):
    """Carga archivos GIS, repara coordenadas y sube a BD manteniendo TODOS los campos."""
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

        status.write(f"✅ Leído: {len(gdf)} registros. Columnas: {list(gdf.columns)}")

        # REPROYECCIÓN OBLIGATORIA A WGS84
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            status.write("🔄 Reproyectando a WGS84 (EPSG:4326)...")
            gdf = gdf.to_crs("EPSG:4326")
        
        # Normalización de columnas
        gdf.columns = [c.lower() for c in gdf.columns]
        
        # Mapeo inteligente (pero conservamos el resto de columnas)
        rename_map = {}
        if 'bocatomas' in nombre_tabla and 'nombre' in gdf.columns: rename_map['nombre'] = 'nom_bocatoma'
        elif 'suelos' in nombre_tabla:
            if 'gridcode' in gdf.columns: rename_map['gridcode'] = 'codigo'
            if 'simbolo' in gdf.columns: rename_map['simbolo'] = 'codigo'
        elif 'zonas_hidrogeologicas' in nombre_tabla and 'nombre' in gdf.columns: 
            rename_map['nombre'] = 'nombre_zona'
            
        if rename_map:
            gdf = gdf.rename(columns=rename_map)

        status.write("📤 Subiendo a Base de Datos (Conservando todos los atributos)...")
        gdf.to_postgis(nombre_tabla, engine, if_exists='replace', index=False)
        
        status.update(label="¡Carga Exitosa!", state="complete", expanded=False)
        st.success(f"Capa **{nombre_tabla}** actualizada. {len(gdf)} registros con {len(gdf.columns)} campos.")
        if len(gdf) > 0: st.balloons()
        
    except Exception as e:
        status.update(label="Error", state="error")
        st.error(f"Error crítico: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def editor_tabla_gis(nombre_tabla, key_editor):
    """Genera un editor de tabla para capas GIS excluyendo la columna de geometría pesada."""
    try:
        # Consultamos columnas excepto 'geometry' para que la tabla sea ligera y legible
        q_cols = text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{nombre_tabla}' AND column_name != 'geometry'")
        cols = pd.read_sql(q_cols, engine)['column_name'].tolist()
        if not cols:
             st.warning(f"La tabla {nombre_tabla} existe pero no tiene columnas legibles.")
             return

        cols_str = ", ".join([f'"{c}"' for c in cols]) # Comillas para nombres seguros
        
        df = pd.read_sql(f"SELECT {cols_str} FROM {nombre_tabla} LIMIT 1000", engine)
        st.info(f"Mostrando primeros 1000 registros de **{nombre_tabla}**. ({len(df.columns)} campos)")
        
        # KEY ÚNICA AQUÍ TAMBIÉN
        df_editado = st.data_editor(df, key=key_editor, use_container_width=True, num_rows="dynamic")
        
        if st.button(f"💾 Guardar Cambios en {nombre_tabla}", key=f"btn_{key_editor}"):
            st.warning("⚠️ Edición directa deshabilitada por seguridad en esta versión. Use la carga de archivos para cambios masivos.")
    except Exception as e:
        st.warning(f"La tabla '{nombre_tabla}' aún no tiene datos o no existe. Cargue un archivo primero.")

# --- 4. INTERFAZ PRINCIPAL ---
st.title("👑 Panel de Administración y Edición de Datos")
st.markdown("---")

tabs = st.tabs([
    "📡 Estaciones", "📊 Índices", "🏠 Predios", "🌊 Cuencas", 
    "🏙️ Municipios", "🌲 Coberturas", "💧 Bocatomas", "⛰️ Hidrogeología", "🌱 Suelos", "🛠️ SQL", "📚 Inventario", "🌧️ Lluvia"
])


# ==============================================================================
# TAB 1: ESTACIONES
# ==============================================================================
with tabs[0]:
    st.header("📡 Gestión de Estaciones")
    sub_editar, sub_crear, sub_carga = st.tabs(["✏️ Editar Existente", "➕ Crear Nueva", "📂 Carga Masiva"])
    
    with sub_editar:
        st.info("Busca una estación para corregir sus coordenadas.")
        if engine:
            with engine.connect() as conn:
                try:
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
                                    nn = c1.text_input("Nombre", value=est_data.get('nom_est', ''))
                                    nm = c1.text_input("Municipio", value=est_data.get('municipio', ''))
                                    nl = c2.number_input("Latitud", value=float(est_data.get('latitud') or 0.0), format="%.5f")
                                    nlo = c2.number_input("Longitud", value=float(est_data.get('longitud') or 0.0), format="%.5f")
                                    if st.form_submit_button("Guardar Cambios"):
                                        conn.execute(text("UPDATE estaciones SET nom_est=:n, municipio=:m, latitud=:l, longitud=:lo WHERE id_estacion=:id"),
                                                    {"n": nn, "m": nm, "l": nl, "lo": nlo, "id": id_sel})
                                        conn.commit()
                                        st.success("Actualizado.")
                                        time.sleep(0.5)
                                        st.rerun()
                except: st.warning("Error cargando lista.")

    with sub_crear:
        with st.form("new_est"):
            c1, c2 = st.columns(2)
            nid = c1.text_input("ID")
            nnom = c1.text_input("Nombre")
            nlat = c2.number_input("Latitud", value=6.0)
            nlon = c2.number_input("Longitud", value=-75.0)
            if st.form_submit_button("Crear"):
                if nid and nnom:
                    with engine.connect() as conn:
                        try:
                            conn.execute(text("INSERT INTO estaciones (id_estacion, nom_est, latitud, longitud) VALUES (:id, :n, :la, :lo)"),
                                        {"id": nid, "n": nnom, "la": nlat, "lo": nlon})
                            conn.commit()
                            st.success("Creada.")
                        except Exception as e: st.error(f"Error: {e}")

    with sub_carga:
        up_meta = st.file_uploader("Cargar 'mapaCVENSO.csv'", type=["csv"], key="up_meta_csv")
        if up_meta and st.button("Procesar", key="btn_proc_meta"):
            try:
                df = pd.read_csv(up_meta, sep=';', encoding='latin-1')
                with engine.connect() as conn:
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
                        except: pass
                    conn.commit()
                st.success("Procesado.")
            except Exception as e: st.error(f"Error: {e}")

# ==============================================================================
# TAB 2: ÍNDICES (FORZANDO PUNTO Y COMA)
# ==============================================================================
with tabs[1]:
    st.header("📊 Índices Climáticos")
    sb1, sb2 = st.tabs(["👁️ Ver Tabla Completa", "📂 Cargar/Actualizar CSV"])
    
    with sb1: # Sub-pestaña: Tabla Completa
        st.markdown("### 📋 Inventario de Predios")
        
        try:
            # 1. DIAGNÓSTICO: ¿Existe la tabla?
            # Intentamos leer solo 1 fila para ver si la tabla responde y qué columnas tiene
            df_test = pd.read_sql('SELECT * FROM predios LIMIT 1', engine)
            
            # Si llegamos aquí, la tabla EXISTE. Ahora filtramos la geometría para que no rompa la tabla visual.
            columnas_todas = df_test.columns.tolist()
            columnas_visibles = [c for c in columnas_todas if c != 'geometry']
            
            if not columnas_visibles:
                st.warning("⚠️ La tabla 'predios' existe, pero solo tiene columna de geometría o está vacía.")
            else:
                # 2. CONSULTA SEGURA: Traemos solo las columnas de texto/números
                # Usamos comillas dobles "{c}" para respetar mayúsculas/minúsculas exactas (ej: "NOMBRE_PRE")
                cols_sql = ", ".join([f'"{c}"' for c in columnas_visibles])
                query = f"SELECT {cols_sql} FROM predios"
                
                df_final = pd.read_sql(query, engine)
                
                st.success(f"✅ Conexión establecida. Se encontraron {len(df_final)} registros.")
                st.dataframe(df_final, use_container_width=True)
                
        except Exception as e:
            # Si falla, mostramos un diagnóstico detallado
            st.error("❌ No se puede visualizar la tabla.")
            st.warning(f"Error técnico: {e}")
            
            st.markdown("---")
            st.info("🔍 **Herramienta de Diagnóstico:**")
            if st.button("Verificar tablas en Base de Datos"):
                try:
                    q_check = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                    df_tablas = pd.read_sql(q_check, engine)
                    st.write("Tablas encontradas en Supabase:", df_tablas)
                except Exception as ex:
                    st.error(f"Ni siquiera pudimos listar las tablas: {ex}")

    with sb2:
        st.markdown("### Cargar Archivo de Índices")
        st.info("Sube 'Indices_Globales.csv'. Se forzará el uso de **punto y coma (;)** como separador.")
        up_i = st.file_uploader("Seleccionar CSV", type=["csv"], key="up_ind_final")
        
        if up_i and st.button("Procesar y Corregir BD", key="btn_load_ind_final"):
            try:
                # LEER CON PUNTO Y COMA EXPLÍCITAMENTE
                df = pd.read_csv(up_i, sep=';', encoding='latin-1', engine='python')
                
                # Limpieza de columnas
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                
                if len(df.columns) < 2:
                    st.error("❌ Error: El archivo no parece estar separado por punto y coma (;). Revisa tu CSV.")
                else:
                    # Guardar en BD reemplazando lo anterior
                    df.to_sql('indices_climaticos', engine, if_exists='replace', index=False)
                    st.success(f"✅ ¡Base de datos corregida! {len(df)} registros con {len(df.columns)} columnas.")
                    st.dataframe(df.head())
                    st.balloons()
            except Exception as e:
                st.error(f"Error: {e}")

# ==============================================================================
# TAB 3: PREDIOS
# ==============================================================================
with tabs[2]:
    st.header("🏠 Gestión de Predios")
    sb1, sb2 = st.tabs(["👁️ Tabla Completa", "📂 Carga GeoJSON"])
    
    with sb1:
        try:
            # Consultamos sin geometría para que sea rápido de ver
            df_p = pd.read_sql("SELECT id_predio, nombre_predio, municipio, area_ha FROM predios LIMIT 2000", engine)
            st.data_editor(df_p, key="ed_pred", use_container_width=True)
        except: st.warning("Sin datos o tabla no existe.")

    with sb2:
        st.info("Carga 'PrediosEjecutados.geojson'. El sistema detectará automáticamente la geometría y reemplazará la tabla existente.")
        
        # Key única para evitar conflictos
        up_gp = st.file_uploader("GeoJSON Predios", type=["geojson", "json"], key="up_pred_geo_smart_v2")
        
        if up_gp:
            try:
                # 1. Leemos CON GEOPANDAS (Para conservar el mapa)
                gdf_p = gpd.read_file(up_gp)
                cols_p = list(gdf_p.columns)
                
                st.markdown("##### 🛠️ Configuración de Columnas")
                c1, c2, c3 = st.columns(3)
                
                # Buscamos columnas candidatas automáticamente
                idx_nom = next((i for i, c in enumerate(cols_p) if c.lower() in ['nombre_pre', 'nombre_predio', 'nombre', 'predio']), 0)
                idx_id = next((i for i, c in enumerate(cols_p) if c.lower() in ['pk_predios', 'id_predio', 'codigo', 'id']), 0)
                idx_mun = next((i for i, c in enumerate(cols_p) if c.lower() in ['nomb_mpio', 'municipio', 'mpio']), 0)
                
                # SELECTORES MANUALES (Tú tienes el control)
                col_nom_p = c1.selectbox("📌 Nombre Predio:", cols_p, index=idx_nom, key="sel_nom_pred")
                col_id_p = c2.selectbox("🔑 ID Único:", cols_p, index=idx_id, key="sel_id_pred")
                col_mun_p = c3.selectbox("🏙️ Municipio:", cols_p, index=idx_mun, key="sel_mun_pred")
                
                if st.button("🚀 Reemplazar y Guardar Predios", key="btn_save_pred_smart"):
                    status = st.status("Procesando...", expanded=True)
                    
                    # 2. Asegurar WGS84
                    if gdf_p.crs and gdf_p.crs.to_string() != "EPSG:4326":
                        status.write("🔄 Reproyectando a WGS84...")
                        gdf_p = gdf_p.to_crs("EPSG:4326")
                    
                    # 3. Renombrar a estándar (nombre_predio, id_predio)
                    # Esto asegura que el visualizador encuentre las columnas
                    gdf_p = gdf_p.rename(columns={
                        col_nom_p: 'nombre_predio',
                        col_id_p: 'id_predio',
                        col_mun_p: 'municipio'
                    })
                    
                    # 4. SUBIDA GEESPACIAL (to_postgis) con 'replace'
                    # 'if_exists="replace"' elimina el error de duplicados borrando lo viejo
                    status.write("📤 Guardando geometría en Base de Datos...")
                    gdf_p.to_postgis('predios', engine, if_exists='replace', index=False)
                    
                    status.update(label="¡Predios Recuperados!", state="complete")
                    st.success(f"✅ Se cargaron {len(gdf_p)} predios con su geometría correctamente.")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error procesando archivo: {e}")


# ==============================================================================
# TAB 4: CUENCAS (CARGADOR PRESERVANDO NOMBRES ORIGINALES EN SELECTOR)
# ==============================================================================
with tabs[3]:
    st.header("🌊 Gestión de Cuencas")
    sb1, sb2 = st.tabs(["👁️ Tabla Maestra", "📂 Carga GeoJSON (Full Data)"])
    
    with sb1:
        try:
            # Consultamos columnas para verificar qué hay en BD
            cols_query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'cuencas' AND column_name != 'geometry'"
            cols_bd = pd.read_sql(cols_query, engine)['column_name'].tolist()
            
            if cols_bd:
                cols_str = ", ".join([f'"{c}"' for c in cols_bd])
                df_c = pd.read_sql(f"SELECT {cols_str} FROM cuencas LIMIT 500", engine)
                st.markdown(f"**Muestra (500 registros):** | **Columnas BD:** {cols_bd}")
                st.dataframe(df_c, use_container_width=True)
            else:
                st.info("La tabla 'cuencas' existe pero no tiene columnas legibles.")
        except: 
            st.warning("No hay datos cargados o la tabla no existe.")

    with sb2:
        st.info("Sube 'SubcuencasAinfluencia.geojson'. Verás los nombres de columna ORIGINALES (ej: N-NSS3).")
        up_c = st.file_uploader("GeoJSON Cuencas", type=["geojson", "json"], key="up_cuen_v4_orig")
        
        if up_c:
            try:
                # 1. Leer archivo SIN TOCAR NOMBRES DE COLUMNAS
                gdf_preview = gpd.read_file(up_c)
                
                # Lista exacta del archivo (Aquí aparecerá 'N-NSS3' con guion)
                cols_originales = list(gdf_preview.columns)
                
                st.success(f"✅ Archivo leído. {len(gdf_preview)} registros.")
                st.write(f"Columnas detectadas: {cols_originales}")
                
                st.markdown("##### 🛠️ Mapeo de Identificadores")
                c1, c2 = st.columns(2)
                
                # Buscamos 'N-NSS3' tal cual, o 'subc_lbl'
                # La búsqueda es insensible a mayúsculas para ayudar, pero el selector muestra el original
                idx_nom = next((i for i, c in enumerate(cols_originales) if c.lower() in ['n-nss3', 'n_nss3', 'subc_lbl', 'nombre']), 0)
                idx_id = next((i for i, c in enumerate(cols_originales) if c.lower() in ['cod', 'objectid', 'id']), 0)
                
                # SELECTORES (Muestran nombre original)
                col_nombre_origen = c1.selectbox("📌 Columna de NOMBRE (Busca N-NSS3):", cols_originales, index=idx_nom, key="sel_cn_nom_orig")
                col_id_origen = c2.selectbox("🔑 Columna de ID (Ej: COD):", cols_originales, index=idx_id, key="sel_cn_id_orig")
                
                if st.button("🚀 Guardar en Base de Datos", key="btn_save_cuen_orig"):
                    status = st.status("Procesando...", expanded=True)
                    
                    # 2. Crear las columnas estándar para la App (nombre_cuenca, id_cuenca)
                    # Tomamos los datos de las columnas que TÚ elegiste
                    gdf_preview['nombre_cuenca'] = gdf_preview[col_nombre_origen].astype(str)
                    gdf_preview['id_cuenca'] = gdf_preview[col_id_origen].astype(str)
                    
                    # 3. AHORA SÍ: Limpieza técnica para SQL (solo al momento de guardar)
                    # Convertimos todo a minúsculas y guiones bajos para que PostGIS no falle
                    # 'N-NSS3' se guardará como 'n_nss3' en la BD, pero sus datos ya están copiados en 'nombre_cuenca'
                    gdf_preview.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in gdf_preview.columns]
                    
                    # 4. Reproyección
                    if gdf_preview.crs and gdf_preview.crs.to_string() != "EPSG:4326":
                        status.write("🔄 Reproyectando a WGS84...")
                        gdf_preview = gdf_preview.to_crs("EPSG:4326")
                    
                    # 5. Guardar
                    status.write("📤 Subiendo a Supabase...")
                    gdf_preview.to_postgis("cuencas", engine, if_exists='replace', index=False)
                    
                    status.update(label="¡Carga Exitosa!", state="complete")
                    st.success(f"✅ Tabla actualizada. Se mapeó **'{col_nombre_origen}'** → **'nombre_cuenca'**.")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error procesando archivo: {e}")


# ==============================================================================
# TAB 5: MUNICIPIOS
# ==============================================================================
with tabs[4]:
    st.header("🏙️ Municipios")
    sb1, sb2 = st.tabs(["👁️ Ver y Editar Tabla", "📂 Cargar GeoJSON"])
    
    with sb1:
        try:
            df_m = pd.read_sql("SELECT * FROM municipios ORDER BY nombre_municipio", engine)
            st.info(f"Gestionando {len(df_m)} municipios.")
            
            # Tabla editable
            df_m_edit = st.data_editor(
                df_m, 
                key="editor_municipios", 
                use_container_width=True,
                height=500
            )
            
            if st.button("💾 Guardar Cambios Municipios", key="btn_save_mun"):
                df_m_edit.to_sql('municipios', engine, if_exists='replace', index=False)
                st.success("✅ Municipios actualizados.")
        except Exception as e:
            st.warning("No hay municipios cargados.")

    with sb2:
        st.info("Carga el archivo de Municipios. Selecciona la columna correcta para evitar el error 'ANTIOQUIA'.")
        up_m = st.file_uploader("GeoJSON Municipios", type=["geojson", "json"], key="up_mun_geo_smart")
        
        if up_m:
            try:
                gdf_m = gpd.read_file(up_m)
                cols_m = list(gdf_m.columns)
                
                st.markdown("##### 🛠️ Mapeo de Columnas")
                c1, c2 = st.columns(2)
                
                # Intentamos adivinar MPIO_CNMBR o NOMBRE_MUNICIPIO
                idx_nom_m = next((i for i, c in enumerate(cols_m) if c.lower() in ['mpio_cnmbr', 'nombre_municipio', 'nombre']), 0)
                idx_cod_m = next((i for i, c in enumerate(cols_m) if c.lower() in ['mpio_cdpmp', 'codigo', 'id_municipio']), 0)
                
                # EL USUARIO ELIGE LA VERDAD
                col_nom_mun = c1.selectbox("📌 Columna NOMBRE MUNICIPIO:", cols_m, index=idx_nom_m, help="Selecciona la que dice 'Medellín', NO la que dice 'Antioquia'")
                col_cod_mun = c2.selectbox("🔑 Columna CÓDIGO DANE:", cols_m, index=idx_cod_m)
                
                if st.button("🚀 Guardar Municipios", key="btn_save_mun_smart"):
                    status = st.status("Procesando...", expanded=True)
                    
                    if gdf_m.crs and gdf_m.crs.to_string() != "EPSG:4326":
                        gdf_m = gdf_m.to_crs("EPSG:4326")
                        
                    # Renombrado Estándar
                    gdf_m = gdf_m.rename(columns={
                        col_nom_mun: 'nombre_municipio', # ESTANDARIZADO
                        col_cod_mun: 'id_municipio'
                    })
                    
                    # Limpieza extra
                    if 'departamento' not in gdf_m.columns:
                        gdf_m['departamento'] = 'Antioquia' # Default si falta
                        
                    gdf_m.to_postgis('municipios', engine, if_exists='replace', index=False)
                    
                    status.update(label="¡Listo!", state="complete")
                    st.success(f"✅ Municipios cargados. Mapeo: **{col_nom_mun}** → **nombre_municipio**")
                    time.sleep(2)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error: {e}")


# ==============================================================================
# TAB 6: GESTIÓN DE RASTERS EN LA NUBE (DEM + COBERTURAS)
# ==============================================================================
with tabs[5]:
    st.header("☁️ Gestión de Rasters (DEM / Coberturas)")
    st.info("Sube aquí los archivos .tif para que el modelo hidrológico los use.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📂 En la Nube")
        rasters = get_raster_list()
        if rasters:
            df_r = pd.DataFrame(rasters)
            if not df_r.empty and 'name' in df_r.columns:
                st.dataframe(df_r[['name', 'created_at']], hide_index=True)
                
                to_del = st.selectbox("Eliminar:", df_r['name'])
                if st.button("🗑️ Borrar Archivo"):
                    ok, msg = delete_raster_from_storage(to_del)
                    if ok: st.success(msg); time.sleep(1); st.rerun()
                    else: st.error(msg)
            else:
                st.info("Bucket vacío o sin acceso.")
        else:
            st.warning("No hay archivos cargados.")

    with col2:
        st.subheader("⬆️ Subir Archivo")
        st.markdown("Requeridos: `DemAntioquia_EPSG3116.tif` y `Cob25m_WGS84.tif`")
        f = st.file_uploader("GeoTIFF", type=["tif", "tiff"], key="up_cloud")
        
        if f:
            if st.button(f"🚀 Subir {f.name} a Supabase"):
                with st.spinner("Subiendo..."):
                    bytes_data = f.getvalue()
                    ok, msg = upload_raster_to_storage(bytes_data, f.name)
                    if ok:
                        st.success(msg)
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(msg)

# ==============================================================================
# TABS 7, 8, 9: GIS ROBUSTO + VISORES DE TABLA (CLAVES ÚNICAS AÑADIDAS)
# ==============================================================================
with tabs[6]: # Bocatomas
    st.header("💧 Bocatomas")
    sb1, sb2 = st.tabs(["👁️ Ver Atributos", "📂 Cargar Archivo"])
    with sb1: editor_tabla_gis("bocatomas", "ed_boca")
    with sb2:
        # AÑADIDA KEY ÚNICA PARA EVITAR ERROR
        f = st.file_uploader("Archivo (ZIP/GeoJSON)", type=["zip", "geojson"], key="up_boca_file")
        if st.button("Cargar", key="btn_load_boca"): cargar_capa_gis_robusta(f, "bocatomas", engine)

with tabs[7]: # Hidro
    st.header("⛰️ Hidrogeología")
    sb1, sb2 = st.tabs(["👁️ Ver Atributos", "📂 Cargar Archivo"])
    with sb1: editor_tabla_gis("zonas_hidrogeologicas", "ed_hidro")
    with sb2:
        # AÑADIDA KEY ÚNICA PARA EVITAR ERROR
        f = st.file_uploader("Archivo (ZIP/GeoJSON)", type=["zip", "geojson"], key="up_hidro_file")
        if st.button("Cargar", key="btn_load_hidro"): cargar_capa_gis_robusta(f, "zonas_hidrogeologicas", engine)

with tabs[8]: # Suelos
    st.header("🌱 Suelos")
    sb1, sb2 = st.tabs(["👁️ Ver Atributos", "📂 Cargar Archivo"])
    with sb1: editor_tabla_gis("suelos", "ed_suelo")
    with sb2:
        # AÑADIDA KEY ÚNICA PARA EVITAR ERROR
        f = st.file_uploader("Archivo (ZIP/GeoJSON)", type=["zip", "geojson"], key="up_suelo_file")
        if st.button("Cargar", key="btn_load_suelo"): cargar_capa_gis_robusta(f, "suelos", engine)

# ==============================================================================
# TAB 10: SQL
# ==============================================================================
with tabs[9]:
    st.header("🛠️ Consola SQL")
    q = st.text_area("Query:")
    if st.button("Ejecutar", key="btn_run_sql"):
        try:
            with engine.connect() as conn:
                if q.strip().lower().startswith("select"):
                    st.dataframe(pd.read_sql(text(q), conn))
                else:
                    conn.execute(text(q))
                    conn.commit()
                    st.success("Hecho.")
        except Exception as e: st.error(str(e))

# ==============================================================================
# TAB 11: INVENTARIO DE ARCHIVOS (NUEVO)
# ==============================================================================
with tabs[10]: # Índice 10 porque es la pestaña número 11 (0-10)
    st.header("📚 Inventario de Archivos del Sistema")
    st.markdown("Documentación técnica de los archivos requeridos para la operación de la plataforma.")
    
    # Definimos la data del inventario manualmente según tu estructura
    inventario_data = [
        {
            "Archivo": "mapaCVENSO.csv",
            "Formato": ".csv",
            "Tipo": "Metadatos Estaciones",
            "Descripción": "Coordenadas, nombres y alturas de las estaciones.",
            "Campos Clave": "Id_estacio, Nom_Est, Latitud_geo, Longitud_geo, alt_est"
        },
        {
            "Archivo": "Indices_Globales.csv",
            "Formato": ".csv",
            "Tipo": "Clima Global",
            "Descripción": "Series históricas de índices macroclimáticos (ONI, SOI, etc).",
            "Campos Clave": "año, mes, anomalia_oni, soi, iod, enso_mes"
        },
        {
            "Archivo": "Predios Ejecutados.geojson",
            "Formato": ".geojson",
            "Tipo": "Vector (Polígonos)",
            "Descripción": "Delimitación de predios intervenidos o gestionados.",
            "Campos Clave": "PK_PREDIOS, NOMBRE_PRE, NOMB_MPIO, AREA_HA"
        },
        {
            "Archivo": "SubcuencasAinfluencia.geojson",
            "Formato": ".geojson",
            "Tipo": "Vector (Polígonos)",
            "Descripción": "Límites hidrográficos y zonas de influencia.",
            "Campos Clave": "COD/OBJECTID, SUBC_LBL, Shape_Area, SZH, AH, ZH"
        },
        {
            "Archivo": "Municipios.geojson",
            "Formato": ".geojson",
            "Tipo": "Vector (Polígonos)",
            "Descripción": "División político-administrativa del departamento.",
            "Campos Clave": "MPIO_CDPMP (Código DANE), MPIO_CNMBR (Nombre)"
        },
        {
            "Archivo": "Cob25m_WGS84.tiff",
            "Formato": ".tiff",
            "Tipo": "Raster",
            "Descripción": "Imagen satelital clasificada de coberturas vegetales.",
            "Campos Clave": "N/A (Valores de píxel: 1=Bosque, 2=Cultivo, etc.)"
        },
        {
            "Archivo": "Bocatomas_Ant.zip",
            "Formato": ".zip (Shapefile)",
            "Tipo": "Vector (Puntos)",
            "Descripción": "Ubicación de captaciones de agua.",
            "Campos Clave": "nombre_bocatoma, caudal, usuario"
        },
        {
            "Archivo": "Zonas_PotHidrogeologico.geojson",
            "Formato": ".geojson",
            "Tipo": "Vector (Polígonos)",
            "Descripción": "Clasificación del potencial de aguas subterráneas.",
            "Campos Clave": "potencial, unidad_geologica"
        },
        {
            "Archivo": "Suelos_Antioquia.geojson",
            "Formato": ".geojson",
            "Tipo": "Vector (Polígonos)",
            "Descripción": "Unidades de suelo y capacidad agrológica.",
            "Campos Clave": "unidad_suelo, textura, grupo_hidro"
        }
    ]
    
    # Crear DataFrame
    df_inv = pd.DataFrame(inventario_data)
    
    # Mostrar tabla bonita
    st.dataframe(
        df_inv,
        column_config={
            "Archivo": st.column_config.TextColumn("Nombre Archivo", width="medium"),
            "Descripción": st.column_config.TextColumn("Descripción", width="large"),
            "Campos Clave": st.column_config.TextColumn("Campos / Columnas", width="large"),
        },
        hide_index=True,
        use_container_width=True
    )


# ==============================================================================
# TAB 12: Precipitación MENSUAL
# ==============================================================================

with tabs[11]:
    st.header("🌧️ Archivo Maestro de Lluvia")
    st.info("Carga el archivo `DatosPptnmes_ENSO.csv`. Se omitirán las celdas vacías (no se rellenan con ceros).")
    
    sb1, sb2 = st.tabs(["👁️ Ver BD", "🚀 Migrar CSV"])
    
    with sb1:
        try:
            df_bd = pd.read_sql("SELECT * FROM precipitacion ORDER BY fecha DESC LIMIT 100", engine)
            st.dataframe(df_bd)
        except: st.warning("No se pudo leer la tabla 'precipitacion'.")

    with sb2:
        up_rain = st.file_uploader("CSV Maestro", type=["csv"])
        if up_rain and st.button("Migrar a BD"):
            try:
                df = pd.read_csv(up_rain)
                # Detección formato ancho (Estaciones en columnas o filas)
                # Asumimos formato matriz: Col 1 = Codigo, Cols 2..N = Fechas
                id_col = df.columns[0]
                fechas = df.columns[1:]
                
                status = st.status("Transformando...", expanded=True)
                
                # Melt: De Ancho a Largo
                df_long = df.melt(id_vars=[id_col], value_vars=fechas, var_name='fecha', value_name='valor')
                df_long = df_long.rename(columns={id_col: 'id_estacion'})
                
                # Conversión
                df_long['fecha'] = pd.to_datetime(df_long['fecha'], errors='coerce')
                df_long['valor'] = pd.to_numeric(df_long['valor'], errors='coerce')
                
                # --- REGLA DE ORO: OMITIR VACÍOS ---
                # Eliminamos filas donde la fecha o el valor sean NaT/NaN.
                # NO llenamos con cero.
                df_final = df_long.dropna(subset=['fecha', 'valor'])
                
                status.write(f"Registros válidos: {len(df_final)}")
                status.write("Subiendo a BD...")
                
                df_final.to_sql('precipitacion', engine, if_exists='replace', index=False, chunksize=5000)
                
                status.update(label="¡Migración Completa!", state="complete")
                st.success("Archivo migrado correctamente respetando vacíos.")
            except Exception as e:
                st.error(f"Error: {e}")
