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
    "🏙️ Municipios", "🌲 Coberturas", "💧 Bocatomas", "⛰️ Hidrogeología", "🌱 Suelos", "🛠️ SQL"
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
# TAB 2: ÍNDICES (RECUPERADO: VER, EDITAR Y CARGAR)
# ==============================================================================
with tabs[1]:
    st.header("📊 Índices Climáticos (ONI, SOI, IOD)")
    sb1, sb2 = st.tabs(["👁️ Ver y Editar Tabla", "📂 Cargar Archivo CSV"])
    
    # SUB-TAB 1: EDITOR DE TABLA
    with sb1:
        try:
            # Traemos todos los datos ordenados por fecha
            df_idx = pd.read_sql("SELECT * FROM indices_climaticos ORDER BY fecha DESC", engine)
            st.markdown(f"**Total Registros:** {len(df_idx)}")
            
            # Editor interactivo
            df_editado = st.data_editor(
                df_idx, 
                key="editor_indices_main", 
                use_container_width=True, 
                num_rows="dynamic",
                height=500
            )
            
            if st.button("💾 Guardar Cambios en Índices", key="btn_save_indices"):
                with st.spinner("Guardando..."):
                    df_editado.to_sql('indices_climaticos', engine, if_exists='replace', index=False)
                    st.success("✅ Tabla actualizada correctamente.")
        except Exception as e:
            st.info("ℹ️ Aún no hay índices cargados o la tabla no existe.")

    # SUB-TAB 2: CARGA CSV
    with sb2:
        st.info("Sube el archivo 'Indices_Globales.csv'.")
        up_i = st.file_uploader("Seleccionar CSV", type=["csv"], key="up_ind_csv_final")
        
        if up_i and st.button("Procesar Carga", key="btn_load_ind_final"):
            try:
                # Intentamos leer con diferentes codificaciones por si acaso
                try:
                    df = pd.read_csv(up_i, encoding='utf-8')
                except:
                    up_i.seek(0)
                    df = pd.read_csv(up_i, encoding='latin-1')
                
                # Limpieza
                df.columns = [c.lower().strip() for c in df.columns]
                if 'id' in df.columns: df = df.drop(columns=['id'])
                
                # Guardar
                df.to_sql('indices_climaticos', engine, if_exists='replace', index=False)
                st.success(f"✅ Se cargaron {len(df)} registros exitosamente.")
                st.balloons()
            except Exception as e: st.error(f"Error al procesar CSV: {e}")

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
# TAB 4: CUENCAS (VISOR COMPLETO)
# ==============================================================================
with tabs[3]:
    st.header("🌊 Gestión de Cuencas")
    sb1, sb2 = st.tabs(["👁️ Tabla Maestra (Todas las Columnas)", "📂 Carga GeoJSON"])
    
    with sb1:
        try:
            # Seleccionamos TODO (*)
            df_c = pd.read_sql("SELECT * FROM cuencas ORDER BY id_cuenca", engine)
            st.caption(f"Mostrando {len(df_c)} cuencas registradas.")
            
            # Editor configurado para mostrar todo
            df_c_ed = st.data_editor(
                df_c, 
                key="ed_cuen_full", 
                use_container_width=True, 
                height=600,
                num_rows="dynamic"
            )
            
            if st.button("💾 Guardar Cambios Cuencas", key="btn_save_cuencas"):
                df_c_ed.to_sql('cuencas', engine, if_exists='replace', index=False)
                st.success("Cambios guardados.")
        except: st.write("Sin datos disponibles.")

    with sb2:
        st.info("Carga 'SubcuencasAinfluencia.geojson'. El sistema corrige duplicados automáticamente.")
        up_c = st.file_uploader("GeoJSON Cuencas", type=["geojson", "json"], key="up_cuen_geo_fix")
        
        if up_c and st.button("Procesar Archivo", key="btn_proc_cuen_fix"):
            try:
                data = json.load(up_c)
                count = 0
                with engine.connect() as conn:
                    for f in data['features']:
                        p = f.get('properties', {})
                        # Lógica segura de mapeo
                        area = float(p.get('Shape_Area', 0))/1_000_000
                        row = {
                            "id": str(p.get('COD', 'SN')),
                            "nom": p.get('SUBC_LBL', 'Sin Nombre'),
                            "area": area,
                            "rio": p.get('SZH', '')
                        }
                        # UPSERT para evitar error de llave duplicada
                        conn.execute(text("""
                            INSERT INTO cuencas (id_cuenca, nombre_cuenca, area_km2, rio_principal)
                            VALUES (:id, :nom, :area, :rio)
                            ON CONFLICT (id_cuenca) DO UPDATE SET 
                            nombre_cuenca = EXCLUDED.nombre_cuenca, area_km2 = EXCLUDED.area_km2
                        """), row)
                        count += 1
                    conn.commit()
                st.success(f"✅ Procesadas {count} cuencas correctamente.")
            except Exception as e: st.error(f"Error: {e}")

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