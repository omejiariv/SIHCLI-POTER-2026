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
    
    with sb1:
        try:
            # Leemos la tabla
            df_idx = pd.read_sql("SELECT * FROM indices_climaticos", engine)
            
            if not df_idx.empty:
                # Verificamos si por error se guardó todo en una columna
                if len(df_idx.columns) < 2:
                    st.warning("⚠️ La tabla actual parece tener formato incorrecto (una sola columna). Por favor ve a la pestaña 'Cargar' y sube el archivo de nuevo para corregirlo.")
                    st.dataframe(df_idx) # Mostramos dataframe simple para diagnosticar
                else:
                    st.success(f"✅ Datos cargados correctamente: {len(df_idx)} registros.")
                    st.data_editor(df_idx, key="ed_idx_final", use_container_width=True, num_rows="dynamic", height=500)
            else:
                st.info("La tabla está vacía.")
        except: st.warning("No hay datos.")

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
            df_p = pd.read_sql("SELECT * FROM predios LIMIT 2000", engine)
            st.data_editor(df_p, key="ed_pred", use_container_width=True)
        except: st.warning("Sin datos.")
    with sb2:
        up_gp = st.file_uploader("GeoJSON", type=["geojson", "json"], key="up_pred_geo")
        if up_gp and st.button("Procesar", key="btn_proc_pred"):
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
                pd.DataFrame(rows).drop_duplicates('id_predio').to_sql('predios', engine, if_exists='append', index=False, method='multi')
                st.success("Cargado.")
            except: st.warning("Error/Duplicados. Use SQL para limpiar.")

# ==============================================================================
# TAB 4: CUENCAS (CORREGIDO: TODOS LOS CAMPOS DEL GEOJSON)
# ==============================================================================
with tabs[3]:
    st.header("🌊 Gestión de Cuencas")
    sb1, sb2 = st.tabs(["👁️ Tabla Maestra (Completa)", "📂 Carga GeoJSON"])
    
    with sb1:
        try:
            # Leemos toda la tabla sin filtrar columnas
            df_c = pd.read_sql("SELECT * FROM cuencas", engine)
            
            # Mostramos estadísticas
            st.markdown(f"**Total Cuencas:** {len(df_c)} | **Total Campos:** {len(df_c.columns)}")
            
            # Editor capaz de mostrar muchas columnas (scroll horizontal automático)
            st.data_editor(
                df_c, 
                key="ed_cuen_full_attribs", 
                use_container_width=True, 
                num_rows="dynamic",
                height=600
            )
        except Exception as e: 
            st.info("No hay datos cargados o la tabla no existe aún.")

    with sb2:
        st.info("Carga 'SubcuencasAinfluencia.geojson'. Se guardarán **TODOS** los atributos (AH, ZH, SZH, etc.)")
        up_c = st.file_uploader("GeoJSON Cuencas", type=["geojson", "json"], key="up_cuen_geo_all_fields")
        
        if up_c and st.button("Procesar Archivo Completo", key="btn_proc_cuen_full"):
            status = st.status("Leyendo archivo geográfico...", expanded=True)
            try:
                # 1. Leer GeoJSON con Geopandas (Captura todos los atributos automáticamente)
                up_c.seek(0)
                gdf = gpd.read_file(up_c)
                
                status.write(f"✅ Archivo leído. {len(gdf)} registros con {len(gdf.columns)} columnas.")
                
                # 2. Estandarización WGS84
                if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
                    status.write("🔄 Reproyectando a WGS84...")
                    gdf = gdf.to_crs("EPSG:4326")
                
                # 3. Normalizar nombres de columnas (minúsculas para evitar problemas en SQL)
                gdf.columns = [c.lower() for c in gdf.columns]
                
                # 4. Mapeo de columnas CLAVE para que la app funcione (sin perder las otras)
                # La app espera 'id_cuenca' y 'nombre_cuenca'. Renombramos las tuyas a estas.
                rename_dict = {}
                
                # Buscamos la columna de código (COD o OBJECTID)
                if 'cod' in gdf.columns: rename_dict['cod'] = 'id_cuenca'
                elif 'objectid' in gdf.columns: rename_dict['objectid'] = 'id_cuenca'
                
                # Buscamos la columna de nombre (SUBC_LBL o N_NSS1)
                if 'subc_lbl' in gdf.columns: rename_dict['subc_lbl'] = 'nombre_cuenca'
                elif 'n_nss1' in gdf.columns: rename_dict['n_nss1'] = 'nombre_cuenca'
                
                # Mapeo de Río Principal
                if 'szh' in gdf.columns: rename_dict['szh'] = 'rio_principal'
                
                # Aplicar cambios de nombre
                gdf = gdf.rename(columns=rename_dict)
                
                # Asegurar que id_cuenca sea texto
                if 'id_cuenca' in gdf.columns:
                    gdf['id_cuenca'] = gdf['id_cuenca'].astype(str)
                
                # 5. SUBIDA COMPLETA (Reemplazo total para reestructurar la tabla con los nuevos campos)
                status.write("📤 Guardando en Base de Datos (Esto incluye AH, ZH, Zona, etc.)...")
                gdf.to_postgis("cuencas", engine, if_exists='replace', index=False)
                
                status.update(label="¡Carga Exitosa!", state="complete", expanded=False)
                st.success(f"✅ Base de datos actualizada con {len(gdf.columns)} campos de información.")
                st.balloons()
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Error detallado: {e}")


# ==============================================================================
# TAB 5: MUNICIPIOS (RECUPERADO)
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
        st.info("Carga el archivo 'Municipios.geojson' para poblar la base de datos.")
        up_m = st.file_uploader("GeoJSON Municipios", type=["geojson", "json"], key="up_mun_geo")
        
        if up_m and st.button("Procesar Municipios", key="btn_proc_mun"):
            try:
                data = json.load(up_m)
                rows = []
                for f in data['features']:
                    p = f.get('properties', {})
                    rows.append({
                        "id_municipio": str(p.get('MPIO_CDPMP', '00000')),
                        "nombre_municipio": p.get('MPIO_CNMBR', ''),
                        "departamento": p.get('DPTO_CNMBR', 'Antioquia'),
                        "poblacion": 0
                    })
                pd.DataFrame(rows).drop_duplicates('id_municipio').to_sql('municipios', engine, if_exists='replace', index=False)
                st.success(f"✅ Cargados {len(rows)} municipios.")
            except Exception as e: st.error(f"Error: {e}")

# ==============================================================================
# TAB 6: COBERTURAS (VER INFO + GUARDAR)
# ==============================================================================
with tabs[5]:
    st.header("🌲 Coberturas Vegetales (Raster .TIFF)")
    
    col_info, col_load = st.columns([1, 2])
    
    # Lógica para mostrar info del archivo actual
    path_cob = "data/coberturas.tif"
    existe = os.path.exists(path_cob)
    
    with col_info:
        st.subheader("ℹ️ Archivo Actual")
        if existe:
            try:
                with rasterio.open(path_cob) as src:
                    st.success("✅ Archivo cargado")
                    st.json({
                        "Ancho (px)": src.width,
                        "Alto (px)": src.height,
                        "Bandas": src.count,
                        "CRS": str(src.crs),
                        "Límites": dict(zip(["Oeste", "Sur", "Este", "Norte"], src.bounds))
                    })
            except Exception as e:
                st.error(f"Archivo corrupto: {e}")
        else:
            st.warning("⚠️ No hay archivo de coberturas en el sistema.")

    with col_load:
        st.subheader("📂 Actualizar / Editar Archivo")
        st.info("Para 'editar' las coberturas, debes subir una nueva versión del archivo **Cob25m_WGS84.tiff**.")
        
        f_tiff = st.file_uploader("Seleccionar nuevo .TIFF", type=["tiff", "tif"], key="up_cob_tif_final")
        
        if f_tiff:
            st.warning("⚠️ Esta acción reemplazará el archivo actual.")
            if st.button("💾 Guardar y Reemplazar", key="btn_save_cob_final"):
                os.makedirs("data", exist_ok=True)
                with open(path_cob, "wb") as f:
                    f.write(f_tiff.getbuffer())
                st.success("✅ Archivo de coberturas actualizado correctamente.")
                time.sleep(1)
                st.rerun()

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
