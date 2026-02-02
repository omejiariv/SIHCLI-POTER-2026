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


# --- PESTAÑA DE CONFIGURACIÓN INICIAL (BLOQUE CORREGIDO) ---
# Pega esto justo después de la línea: tabs = st.tabs([...])

st.markdown("### 🛠️ Zona de Peligro: Reinicio del Sistema")
with st.expander("Mostrar Controles de Reinicio de Base de Datos", expanded=True):
    st.warning("⚠️ ESTA ACCIÓN ES IRREVERSIBLE. BORRARÁ TODOS LOS DATOS.")
    
    # HE CAMBIADO EL NOMBRE DEL BOTÓN PARA FORZAR LA ACTUALIZACIÓN
    if st.button("🔥 EJECUTAR REINICIO TOTAL (CASCADE) 🔥", key="btn_nuke_v3"):
        try:
            with engine.begin() as conn:
                st.write("⏳ Iniciando secuencia de borrado...")
                
                # 1. BORRADO EN ORDEN INVERSO (Hijos primero, luego Padres)
                # Usamos CASCADE en todo por seguridad
                conn.execute(text("DROP TABLE IF EXISTS precipitacion CASCADE;"))
                conn.execute(text("DROP TABLE IF EXISTS indices_climaticos CASCADE;"))
                conn.execute(text("DROP TABLE IF EXISTS estaciones CASCADE;"))
                
                st.write("✅ Tablas eliminadas. Creando nueva estructura...")
                
                # 2. CREACIÓN DE TABLAS
                # Estaciones (Padre)
                conn.execute(text("""
                    CREATE TABLE estaciones (
                        id_estacion TEXT PRIMARY KEY,
                        nombre TEXT,
                        longitud FLOAT,
                        latitud FLOAT,
                        altitud FLOAT,
                        municipio TEXT,
                        departamento TEXT,
                        subregion TEXT,
                        corriente TEXT
                    );
                """))
                
                # Índices
                conn.execute(text("""
                    CREATE TABLE indices_climaticos (
                        fecha DATE PRIMARY KEY,
                        enso_año TEXT,
                        enso_mes TEXT,
                        anomalia_oni FLOAT,
                        temp_sst FLOAT,
                        temp_media FLOAT,
                        soi FLOAT,
                        iod FLOAT,
                        fase_enso TEXT
                    );
                """))
                
                # Precipitacion (Hija)
                conn.execute(text("""
                    CREATE TABLE precipitacion (
                        fecha DATE,
                        id_estacion TEXT,
                        valor FLOAT,
                        origen TEXT,
                        PRIMARY KEY (fecha, id_estacion),
                        CONSTRAINT fk_estacion FOREIGN KEY (id_estacion) REFERENCES estaciones(id_estacion)
                    );
                    CREATE INDEX idx_precip_fecha ON precipitacion(fecha);
                    CREATE INDEX idx_precip_estacion ON precipitacion(id_estacion);
                """))
                
            st.success("✅ ¡BASE DE DATOS REINICIADA CORRECTAMENTE!")
            st.balloons()
            time.sleep(2)
            st.rerun() # Recarga la página automáticamente
            
        except Exception as e:
            st.error(f"❌ Error crítico: {e}")



# ==============================================================================
# TAB 0: GESTIÓN DE ESTACIONES (CORREGIDO)
# ==============================================================================
with tabs[0]: 
    st.header("📍 Gestión de Estaciones")
    
    subtab_ver, subtab_carga = st.tabs(["👁️ Editor de Catálogo", "📂 Carga Masiva (CSV)"])
    
    # --- SUB-PESTAÑA 1: VER Y EDITAR ---
    with subtab_ver:
        st.info("Edita las propiedades directamente en la tabla y guarda los cambios.")
        
        if st.button("🔄 Refrescar Tabla Estaciones"):
            st.cache_data.clear()
            st.rerun()
            
        try:
            # 1. Traemos todas las estaciones
            df_est_db = pd.read_sql("SELECT * FROM estaciones ORDER BY id_estacion", engine)
            
            # 2. EDITOR DE DATOS INTERACTIVO
            df_editado = st.data_editor(
                df_est_db,
                num_rows="dynamic",
                key="editor_estaciones",
                use_container_width=True,
                column_config={
                    "id_estacion": st.column_config.TextColumn("Código", disabled=True),
                    "nombre": "Nombre",
                    "municipio": "Municipio",
                    "latitud": st.column_config.NumberColumn("Latitud", format="%.6f"),
                    "longitud": st.column_config.NumberColumn("Longitud", format="%.6f")
                }
            )
            
            # 3. BOTÓN GUARDAR (LÓGICA BLINDADA CON UPSERT)
            if st.button("💾 Guardar Cambios en Catálogo"):
                with st.spinner("Sincronizando cambios de forma segura..."):
                    try:
                        with engine.begin() as conn:
                            # A. Subimos a tabla temporal
                            df_editado.to_sql('temp_est_edit', conn, if_exists='replace', index=False)
                            
                            # B. Ejecutamos UPSERT (Actualizar o Insertar)
                            conn.execute(text("""
                                INSERT INTO estaciones (id_estacion, nombre, latitud, longitud, altitud, municipio, departamento, subregion, corriente)
                                SELECT id_estacion, nombre, latitud, longitud, altitud, municipio, departamento, subregion, corriente
                                FROM temp_est_edit
                                ON CONFLICT (id_estacion) DO UPDATE SET
                                    nombre = EXCLUDED.nombre,
                                    latitud = EXCLUDED.latitud,
                                    longitud = EXCLUDED.longitud,
                                    altitud = EXCLUDED.altitud,
                                    municipio = EXCLUDED.municipio,
                                    departamento = EXCLUDED.departamento,
                                    subregion = EXCLUDED.subregion,
                                    corriente = EXCLUDED.corriente;
                            """))
                            
                            # C. Limpieza
                            conn.execute(text("DROP TABLE IF EXISTS temp_est_edit"))
                            
                        st.success("✅ Catálogo actualizado correctamente (Sin romper vínculos).")
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error("❌ No se pudo guardar.")
                        st.warning(f"Detalle técnico: {e}")

        # ⬇️ ESTE ES EL BLOQUE QUE TE FALTABA ⬇️
        except Exception as e:
            st.warning("No se pudo cargar el catálogo. ¿Quizás está vacío?")
            st.error(f"Error de carga: {e}")

    # --- SUB-PESTAÑA 2: CARGA MASIVA ---
    with subtab_carga:
        st.write("Sube `mapaCVENSO.csv` limpio.")
        up_est = st.file_uploader("Cargar CSV Estaciones", type=["csv"], key="up_est_csv_corrected_final")
        
        if up_est:
            if st.button("🚀 Cargar Catálogo"):
                try:
                    df_new = pd.read_csv(up_est, sep=';', decimal=',')
                    df_new.columns = df_new.columns.str.lower().str.strip()
                    
                    rename_map = {'id_estacio': 'id_estacion', 'nom_est': 'nombre', 'longitud_geo': 'longitud', 'latitud_geo': 'latitud', 'alt_est': 'altitud'}
                    df_new = df_new.rename(columns={k: v for k, v in rename_map.items() if k in df_new.columns})
                    
                    cols_validas = ['id_estacion', 'nombre', 'longitud', 'latitud', 'altitud', 'municipio', 'departamento', 'subregion', 'corriente']
                    df_final = df_new[[c for c in df_new.columns if c in cols_validas]]

                    for c in ['longitud', 'latitud', 'altitud']:
                        if c in df_final.columns:
                            df_final[c] = pd.to_numeric(df_final[c].astype(str).str.replace(',', '.'), errors='coerce')

                    # UPSERT manual para carga masiva (seguridad extra)
                    df_final.to_sql('temp_est_load', engine, if_exists='replace', index=False)
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO estaciones (id_estacion, nombre, latitud, longitud, altitud, municipio, departamento, subregion, corriente)
                            SELECT id_estacion, nombre, latitud, longitud, altitud, municipio, departamento, subregion, corriente
                            FROM temp_est_load
                            ON CONFLICT (id_estacion) DO UPDATE SET
                                nombre = EXCLUDED.nombre,
                                latitud = EXCLUDED.latitud,
                                longitud = EXCLUDED.longitud,
                                altitud = EXCLUDED.altitud;
                        """))
                        conn.execute(text("DROP TABLE IF EXISTS temp_est_load"))

                    st.success(f"✅ Catálogo cargado: {len(df_final)} estaciones.")
                    st.balloons()
                except Exception as ex:
                    st.error(f"Error cargando estaciones: {ex}")


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
    st.info("Aquí administras la capa base de predios (Catastro).")

    sb1, sb2 = st.tabs(["👁️ Tabla Completa", "📂 Carga GeoJSON"])

    # --- SUB-PESTAÑA 1: VISUALIZAR ---
    with sb1:
        try:
            # 1. Leemos la tabla cruda sin filtros
            query_check = "SELECT * FROM predios LIMIT 5"
            df_preview = pd.read_sql(query_check, engine)
            
            # Si no da error, traemos todo (excluyendo geometría pesada)
            cols = [c for c in df_preview.columns if c != 'geometry']
            cols_sql = ", ".join([f'"{c}"' for c in cols]) # Protegemos nombres
            
            df_predios = pd.read_sql(f"SELECT {cols_sql} FROM predios", engine)
            
            st.success(f"✅ Se encontraron {len(df_predios)} predios en la base de datos.")
            st.dataframe(df_predios, use_container_width=True)
            
        except Exception as e:
            st.warning("No se pudo leer la tabla 'predios'. Posiblemente aún no se ha cargado correctamente.")
            st.error(f"Detalle técnico: {e}")

    # --- SUB-PESTAÑA 2: CARGAR (AQUÍ ESTÁ LA MAGIA) ---
    with sb2:
        st.write("Sube el archivo `PrediosEjecutados.geojson`.")
        up_file = st.file_uploader("GeoJSON Predios", type=["geojson", "json"], key="up_pred")
        
        if up_file:
            if st.button("🚀 Reemplazar Base de Datos de Predios"):
                with st.spinner("Procesando geometría y normalizando datos..."):
                    try:
                        # 1. Leer el archivo
                        import geopandas as gpd
                        gdf = gpd.read_file(up_file)
                        
                        # 2. NORMALIZACIÓN (La Clave del Éxito)
                        # Convertimos todos los nombres de columnas a minúsculas para evitar conflictos SQL
                        gdf.columns = map(str.lower, gdf.columns)
                        
                        # 3. Verificar y corregir proyección
                        if gdf.crs is None:
                            gdf.set_crs(epsg=4326, inplace=True)
                        else:
                            gdf = gdf.to_crs(epsg=4326)
                            
                        # 4. Limpieza de geometrías
                        # Convertimos MultiPolygon a Polygon si es necesario o arreglamos geometrías inválidas
                        gdf['geometry'] = gdf.geometry.buffer(0) 
                        
                        # 5. SUBIDA A SUPABASE (PostGIS)
                        # if_exists='replace' BORRA lo anterior y crea la tabla nueva limpia
                        gdf.to_postgis("predios", engine, if_exists='replace', index=False)
                        
                        st.success("✅ ¡Carga Exitosa! La tabla 'predios' ha sido creada correctamente.")
                        st.balloons()
                        
                        # Mostrar resumen de lo que se subió
                        st.write("Resumen de columnas creadas (Minúsculas):")
                        st.write(list(gdf.columns))
                        
                    except Exception as e:
                        st.error(f"❌ Error crítico subiendo predios: {e}")


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
            "Campos Clave": "id_estacion, nombre, latitud, longitud, altitud"
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
# ==============================================================================
# TAB 11: LLUVIA (BLOQUE FINAL CORREGIDO)
# ==============================================================================
with tabs[11]:
    st.header("🌧️ Gestión de Lluvia e Índices")
    
    # 1. DEFINICIÓN DE SUB-PESTAÑAS (ESTA ES LA CLAVE QUE FALTABA)
    t_explorar, t_carga = st.tabs(["🔍 Explorar y Editar Datos", "📂 Carga Masiva (Matriz)"])
    
    # --- SUB-PESTAÑA 1: EXPLORADOR ---
    with t_explorar:
        st.info("Edición puntual de datos por estación y año.")
        try:
            # Selector de Estación
            estaciones_list = pd.read_sql("SELECT id_estacion, nombre FROM estaciones", engine)
            
            if estaciones_list.empty:
                st.warning("⚠️ Primero debes cargar el catálogo de estaciones.")
            else:
                opciones = estaciones_list.apply(lambda x: f"{x['id_estacion']} - {x['nombre']}", axis=1)
                sel_est = st.selectbox("Selecciona Estación:", opciones)
                
                if sel_est:
                    cod_est = sel_est.split(" - ")[0]
                    
                    # --- MEJORA: Consultar años disponibles primero ---
                    q_years = text(f"""
                        SELECT DISTINCT EXTRACT(YEAR FROM fecha)::int as anio 
                        FROM precipitacion 
                        WHERE id_estacion = '{cod_est}' 
                        ORDER BY anio DESC
                    """)
                    df_years = pd.read_sql(q_years, engine)
                    
                    if df_years.empty:
                        st.error("⚠️ Esta estación NO tiene datos de lluvia cargados en la base de datos.")
                        # Opción para añadir datos manualmente desde cero
                        anios_disp = range(2026, 1969, -1)
                    else:
                        st.success(f"✅ Datos disponibles: {len(df_years)} años registrados ({df_years['anio'].min()} - {df_years['anio'].max()})")
                        anios_disp = df_years['anio'].tolist()

                    # Selector de Año Inteligente
                    anio_sel = st.selectbox("Selecciona Año:", anios_disp)
                    
                    # Consulta de datos (usando nombres nuevos)
                    query_data = text(f"""
                        SELECT fecha, valor, origen 
                        FROM precipitacion 
                        WHERE id_estacion = '{cod_est}' 
                        AND EXTRACT(YEAR FROM fecha) = {anio_sel}
                        ORDER BY fecha ASC
                    """)
                    df_lluvia_est = pd.read_sql(query_data, engine)
                    
                    
                    col_edit, col_chart = st.columns([1, 2])
                    
                    with col_edit:
                        st.write(f"**Datos:** {cod_est} - {anio_sel}")
                        if df_lluvia_est.empty:
                            st.warning("No hay registros.")
                            # Creamos un DF vacío con estructura para permitir agregar
                            df_lluvia_est = pd.DataFrame(columns=['fecha', 'valor', 'origen'])
                        
                        df_edited = st.data_editor(
                            df_lluvia_est,
                            num_rows="dynamic",
                            key=f"ed_rain_{cod_est}_{anio_sel}",
                            column_config={
                                "fecha": st.column_config.DateColumn("Fecha", format="YYYY-MM-DD"),
                                "valor": st.column_config.NumberColumn("Precipitación (mm)"),
                                "origen": st.column_config.SelectboxColumn("Origen", options=["real", "interpolado", "editado"])
                            }
                        )
                        
                        if st.button("💾 Guardar Cambios"):
                            if not df_edited.empty:
                                with engine.begin() as conn:
                                    # Borrar datos viejos de ese año/estación
                                    conn.execute(text(f"DELETE FROM precipitacion WHERE id_estacion='{cod_est}' AND EXTRACT(YEAR FROM fecha)={anio_sel}"))
                                    # Insertar nuevos
                                    df_edited['id_estacion'] = cod_est
                                    df_edited.to_sql('precipitacion', engine, if_exists='append', index=False)
                                st.success("✅ Datos guardados.")
                                time.sleep(0.5)
                                st.rerun()

                    with col_chart:
                        if not df_edited.empty and 'fecha' in df_edited.columns and 'valor' in df_edited.columns:
                            st.line_chart(df_edited.set_index('fecha')['valor'])

        except Exception as e:
            st.error(f"Error cargando explorador: {e}")

    # --- SUB-PESTAÑA 2: CARGA MASIVA ---
    with t_carga:
        st.write("Sube `DatosPptnmes_ENSO.csv`.")
        up_rain = st.file_uploader("Cargar Matriz de Lluvia", type=["csv"], key="up_rain_final_fix")
        
        if up_rain:
            if st.button("🚀 Procesar y Cargar Lluvia"):
                status = st.status("Iniciando carga masiva...", expanded=True)
                try:
                    # 1. Leer y Limpiar
                    df = pd.read_csv(up_rain, sep=';', decimal=',')
                    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
                    df = df.dropna(subset=['fecha'])
                    
                    # 2. Detector de Años Mutantes (Ej: 20200)
                    max_year = df['fecha'].dt.year.max()
                    if max_year > 2100:
                        status.update(label="⚠️ ERROR CRÍTICO DE FECHAS", state="error")
                        st.error(f"❌ Se detectaron años incorrectos (Ej: {max_year}). Revisa el CSV (posibles 20200).")
                        st.stop()
                    
                    status.write("Transformando datos (Pivot)...")
                    # 3. Transformación (Melt)
                    est_cols = [c for c in df.columns if c != 'fecha']
                    df_long = df.melt(id_vars=['fecha'], value_vars=est_cols, var_name='id_estacion', value_name='valor')
                    df_long['valor'] = pd.to_numeric(df_long['valor'], errors='coerce')
                    df_long = df_long.dropna(subset=['valor'])
                    df_long['origen'] = 'real'
                    
                    # 4. Carga por Lotes con Salvavidas
                    total = len(df_long)
                    status.write(f"Cargando {total:,.0f} registros...")
                    
                    chunk_size = 20000
                    total_chunks = (total // chunk_size) + 1
                    bar = status.progress(0)
                    
                    for i, start in enumerate(range(0, total, chunk_size)):
                        batch = df_long.iloc[start : start + chunk_size]
                        
                        # Tabla temporal
                        batch.to_sql('temp_rain_load', engine, if_exists='replace', index=False)
                        
                        with engine.begin() as conn:
                            # A. SALVAVIDAS: Crear estaciones fantasmas si faltan en el catálogo
                            # Esto evita el error de Foreign Key
                            conn.execute(text("""
                                INSERT INTO estaciones (id_estacion, nombre) 
                                SELECT DISTINCT id_estacion, 'Auto-Generada ' || id_estacion 
                                FROM temp_rain_load 
                                WHERE id_estacion NOT IN (SELECT id_estacion FROM estaciones)
                            """))
                            
                            # B. UPSERT: Insertar o Actualizar lluvia
                            conn.execute(text("""
                                INSERT INTO precipitacion (fecha, id_estacion, valor, origen) 
                                SELECT fecha, id_estacion, valor, origen FROM temp_rain_load 
                                ON CONFLICT (fecha, id_estacion) 
                                DO UPDATE SET valor = EXCLUDED.valor
                            """))
                        
                        bar.progress((i + 1) / total_chunks)
                    
                    status.update(label="✅ ¡Carga Completa!", state="complete")
                    st.balloons()
                    
                except Exception as ex:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Detalle técnico: {ex}")