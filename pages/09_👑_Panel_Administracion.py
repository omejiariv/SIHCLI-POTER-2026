# pages/09_👑_Panel_Administracion.py


import streamlit as st
import pandas as pd
import json  # Para leer archivos GeoJSON/JSON
import io    # Para manejo de buffers de archivos
import time  # Para pausas o efectos visuales
from sqlalchemy import text  # Para escribir consultas SQL
from modules.database import get_engine  # Tu conexión centralizada a la BD


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

# Definición de Pestañas
tab_est, tab_indices, tab_predios, tab_cuencas, tab_sql = st.tabs([
    "🌧️ Estaciones & Lluvias",
    "📉 Índices Globales", 
    "🏡 Predios (Fincas)",
    "🌊 Cuencas",
    "🛠️ Consola SQL"
])



# ==============================================================================
# ====================================================================
# TAB 1: GESTIÓN DE ESTACIONES (EDICIÓN + CREACIÓN + CARGA)
# ====================================================================
with tab_est:
    st.header("🌧️ Gestión de Estaciones Hidroclimáticas")
    
    # Sub-pestañas internas para organizar mejor
    sub_editar, sub_crear, sub_carga = st.tabs([
        "✏️ Editar Existente", 
        "➕ Crear Nueva", 
        "📥 Carga Masiva (Históricos)"
    ])

    # ----------------------------------------------------------------
    # SUB-PESTAÑA 1: EDITAR ESTACIÓN (CORREGIDO)
    # ----------------------------------------------------------------
    with sub_editar:
        st.info("Busca una estación para corregir sus coordenadas, nombre o metadatos.")
        
        # 1. Cargar lista de estaciones para el buscador
        engine = get_engine()
        if engine:
            try:
                with engine.connect() as conn:
                    # Traemos solo ID y Nombre para el buscador (liviano)
                    query_list = text("SELECT id_estacion, nom_est FROM estaciones ORDER BY nom_est")
                    df_lista = pd.read_sql(query_list, conn)
                    
                    # Crear lista de opciones: "Nombre (ID)"
                    df_lista['display'] = df_lista['nom_est'] + " (" + df_lista['id_estacion'].astype(str) + ")"
                    opciones = df_lista['display'].tolist()
                    
                    # Selector
                    seleccion = st.selectbox("🔍 Buscar Estación:", opciones, index=None, placeholder="Escribe para buscar...")
                    
                    if seleccion:
                        # Extraer ID del string seleccionado
                        id_sel = seleccion.split('(')[-1].replace(')', '')
                        
                        # 2. Cargar datos completos de la estación seleccionada
                        query_full = text("SELECT * FROM estaciones WHERE id_estacion = :id")
                        df_full = pd.read_sql(query_full, conn, params={"id": id_sel})
                        
                        # --- CORRECCIÓN CLAVE: NORMALIZAR COLUMNAS ---
                        # Convertimos todo a minúsculas para evitar error por 'Latitud' vs 'latitud'
                        df_full.columns = [c.lower() for c in df_full.columns]
                        
                        if not df_full.empty:
                            est_data = df_full.iloc[0]
                            
                            st.divider()
                            st.markdown(f"### 📝 Editando: **{est_data.get('nom_est', 'Sin Nombre')}**")
                            
                            # Formulario de Edición
                            with st.form("form_editar_estacion"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    # Usamos .get() para mayor seguridad si falta algún campo
                                    new_nom = st.text_input("Nombre Estación", value=est_data.get('nom_est', ''))
                                    
                                    # Selectboxes con validación de valor actual
                                    cat_actual = est_data.get('categoria', 'Pluviométrica')
                                    opciones_cat = ["Pluviométrica", "Limnimétrica", "Climática", "Otras"]
                                    index_cat = opciones_cat.index(cat_actual) if cat_actual in opciones_cat else 0
                                    new_cat = st.selectbox("Categoría", opciones_cat, index=index_cat)
                                    
                                    tec_actual = est_data.get('tecnologia', 'Convencional')
                                    opciones_tec = ["Convencional", "Automática", "Radar"]
                                    index_tec = opciones_tec.index(tec_actual) if tec_actual in opciones_tec else 0
                                    new_tec = st.selectbox("Tecnología", opciones_tec, index=index_tec)
                                    
                                    new_mun = st.text_input("Municipio", value=est_data.get('municipio', '') or "")
                                
                                with col2:
                                    # Conversión segura a float (maneja None/Null de base de datos)
                                    def safe_float(val):
                                        try:
                                            return float(val) if val is not None else 0.0
                                        except:
                                            return 0.0

                                    new_lat = st.number_input("Latitud (Decimal)", value=safe_float(est_data.get('latitud')), format="%.5f")
                                    new_lon = st.number_input("Longitud (Decimal)", value=safe_float(est_data.get('longitud')), format="%.5f")
                                    new_elev = st.number_input("Elevación (msnm)", value=safe_float(est_data.get('elevacion')))
                                    
                                    # El ID lo mostramos pero bloqueado
                                    new_cod = st.text_input("Código (ID)", value=est_data.get('id_estacion', ''), disabled=True, help="El ID no se puede cambiar.")

                                # El botón está DENTRO del form (Indentación correcta)
                                submitted = st.form_submit_button("💾 Guardar Cambios")
                                
                                if submitted:
                                    try:
                                        # Query de Actualización
                                        update_q = text("""
                                            UPDATE estaciones 
                                            SET nom_est=:nom, categoria=:cat, tecnologia=:tec, 
                                                municipio=:mun, latitud=:lat, longitud=:lon, elevacion=:elev
                                            WHERE id_estacion=:id
                                        """)
                                        conn.execute(update_q, {
                                            "nom": new_nom, "cat": new_cat, "tec": new_tec,
                                            "mun": new_mun, "lat": new_lat, "lon": new_lon,
                                            "elev": new_elev, "id": id_sel
                                        })
                                        conn.commit()
                                        st.success(f"✅ Estación '{new_nom}' actualizada correctamente.")
                                        st.rerun() 
                                    except Exception as e:
                                        st.error(f"Error al actualizar: {e}")
            except Exception as e:
                st.error(f"Error de conexión: {e}")

    # ----------------------------------------------------------------
    # SUB-PESTAÑA 2: CREAR NUEVA ESTACIÓN
    # ----------------------------------------------------------------
    with sub_crear:
        st.markdown("### ➕ Registrar Nueva Estación")
        with st.form("form_crear_estacion"):
            c1, c2 = st.columns(2)
            with c1:
                new_id = st.text_input("Código ID (Único)", placeholder="Ej: 12045010")
                new_nom = st.text_input("Nombre Estación", placeholder="Ej: Hacienda La Esperanza")
                new_mun = st.text_input("Municipio", placeholder="Ej: Rionegro")
            with c2:
                new_lat = st.number_input("Latitud", format="%.5f", value=6.0)
                new_lon = st.number_input("Longitud", format="%.5f", value=-75.0)
                new_elev = st.number_input("Elevación", value=1500.0)
            
            btn_crear = st.form_submit_button("🚀 Crear Estación")
            
            if btn_crear:
                if new_id and new_nom:
                    engine = get_engine()
                    if engine:
                        try:
                            with engine.connect() as conn:
                                insert_q = text("""
                                    INSERT INTO estaciones (id_estacion, nom_est, municipio, latitud, longitud, elevacion)
                                    VALUES (:id, :nom, :mun, :lat, :lon, :elev)
                                """)
                                conn.execute(insert_q, {
                                    "id": new_id, "nom": new_nom, "mun": new_mun,
                                    "lat": new_lat, "lon": new_lon, "elev": new_elev
                                })
                                conn.commit()
                                st.success("✅ Estación creada exitosamente.")
                        except Exception as e:
                            st.error(f"Error creando estación: {e}")
                else:
                    st.warning("⚠️ El ID y el Nombre son obligatorios.")

    # ----------------------------------------------------------------
    # SUB-PESTAÑA 3: CARGA DE METADATOS (mapaCVENSO.csv)
    # ----------------------------------------------------------------
    with sub_carga:
        st.markdown("### 📥 Carga de Metadatos (Coordenadas y Detalles)")
        st.info("Sube aquí el archivo **mapaCVENSO.csv**. El sistema actualizará las coordenadas y nombres de las estaciones existentes sin borrar sus lluvias.")

        uploaded_meta = st.file_uploader("Arrastra el archivo mapaCVENSO.csv", type=["csv"], key="meta_upload")
        
        if uploaded_meta:
            if st.button("🚀 Procesar y Actualizar Metadatos"):
                engine = get_engine()
                if engine:
                    with st.spinner("Leyendo archivo y actualizando base de datos..."):
                        try:
                            # 1. Leer CSV (Detectando separador punto y coma ';')
                            df_meta = pd.read_csv(uploaded_meta, sep=';', encoding='latin-1', engine='python')
                            
                            # Limpieza de nombres de columnas (quitar espacios)
                            df_meta.columns = [c.strip() for c in df_meta.columns]
                            
                            # 2. Verificar columnas críticas
                            cols_necesarias = ['Id_estacio', 'Nom_Est', 'Latitud_geo', 'Longitud_geo', 'alt_est']
                            if not all(col in df_meta.columns for col in cols_necesarias):
                                st.error(f"❌ Faltan columnas clave. Se esperan: {cols_necesarias}")
                                st.write("Columnas encontradas:", df_meta.columns.tolist())
                            else:
                                count_updated = 0
                                count_inserted = 0
                                
                                with engine.connect() as conn:
                                    # 3. Iterar y hacer UPSERT (Insertar o Actualizar)
                                    # Es más lento que bulk insert, pero seguro para no romper FKs
                                    for _, row in df_meta.iterrows():
                                        try:
                                            # Mapeo de valores (seguro contra NaNs)
                                            s_id = str(row['Id_estacio']).strip()
                                            s_nom = str(row['Nom_Est']).strip()
                                            s_mun = str(row['municipio']).strip() if 'municipio' in df_meta.columns else None
                                            
                                            # Convertir coordenadas (reemplazar coma por punto si es necesario)
                                            def clean_float(val):
                                                if pd.isna(val): return 0.0
                                                if isinstance(val, str):
                                                    val = val.replace(',', '.')
                                                try:
                                                    return float(val)
                                                except:
                                                    return 0.0

                                            s_lat = clean_float(row['Latitud_geo'])
                                            s_lon = clean_float(row['Longitud_geo'])
                                            s_alt = clean_float(row['alt_est'])

                                            # Query UPSERT (PostgreSQL syntax)
                                            # Intenta insertar, si hay conflicto de ID, actualiza los campos
                                            upsert_query = text("""
                                                INSERT INTO estaciones (id_estacion, nom_est, municipio, latitud, longitud, elevacion)
                                                VALUES (:id, :nom, :mun, :lat, :lon, :elev)
                                                ON CONFLICT (id_estacion) 
                                                DO UPDATE SET 
                                                    nom_est = EXCLUDED.nom_est,
                                                    municipio = EXCLUDED.municipio,
                                                    latitud = EXCLUDED.latitud,
                                                    longitud = EXCLUDED.longitud,
                                                    elevacion = EXCLUDED.elevacion;
                                            """)
                                            
                                            conn.execute(upsert_query, {
                                                "id": s_id, "nom": s_nom, "mun": s_mun,
                                                "lat": s_lat, "lon": s_lon, "elev": s_alt
                                            })
                                            
                                            # Nota: No podemos saber fácilmente si fue insert o update sin lógica compleja,
                                            # pero asumimos éxito si no falla.
                                            count_updated += 1
                                            
                                        except Exception as row_ex:
                                            print(f"Error en fila {row['Id_estacio']}: {row_ex}")
                                    
                                    conn.commit()
                                    st.success(f"✅ ¡Proceso finalizado! Se procesaron {count_updated} estaciones.")
                                    st.balloons()
                                    
                        except Exception as e:
                            st.error(f"Error procesando el archivo: {e}")


# --- PESTAÑA 2: GESTIÓN DE ÍNDICES GLOBALES ---
with tab_indices:
    st.header("📉 Gestión de Índices Climáticos (ONI, SOI, IOD)")
    st.info("Sube aquí el archivo 'Indices_Globales_1970_2024.csv' limpio (con puntos decimales).")
    
    uploaded_idx = st.file_uploader("Seleccionar CSV de Índices", type=["csv"], key="idx_uploader")
    
    if uploaded_idx:
        try:
            # INTENTO 1: Leer como UTF-8 (Estándar web)
            try:
                df_idx = pd.read_csv(uploaded_idx, sep=None, engine='python', encoding='utf-8')
            except UnicodeDecodeError:
                # INTENTO 2: Si falla por la 'ñ', leer como Latin-1 (Estándar Excel)
                uploaded_idx.seek(0) # Rebobinar el archivo al principio
                df_idx = pd.read_csv(uploaded_idx, sep=None, engine='python', encoding='latin-1')
            
            st.write("Vista Previa de los Datos:", df_idx.head())
            
            # Validación básica (convertimos nombres de columnas a minúsculas para comparar)
            df_idx.columns = [c.lower().strip() for c in df_idx.columns]
            cols_esperadas = ['anomalia_oni', 'soi', 'iod']
            
            # Verificamos si al menos una de las columnas clave existe
            if not any(col in df_idx.columns for col in cols_esperadas):
                st.error(f"❌ El archivo no parece contener índices climáticos. Se esperan columnas como: {cols_esperadas}")
            else:
                if st.button("🚀 Cargar a Base de Datos (Sobreescribir)"):
                    engine = get_engine()
                    if engine:
                        with st.spinner("Cargando índices..."):
                            try:
                                with engine.connect() as conn:
                                    # 1. Eliminar columna 'id' si existe (dejemos que la BD ponga sus propios IDs)
                                    if 'id' in df_idx.columns:
                                        df_idx = df_idx.drop(columns=['id'])
                                    
                                    # 2. Insertar datos (append)
                                    df_idx.to_sql('indices_climaticos', con=conn, if_exists='replace', index=False)
                                    
                                    st.success(f"✅ ¡Éxito! Se han cargado {len(df_idx)} registros históricos.")
                                    st.balloons()
                            except Exception as e:
                                st.error(f"Error en la carga: {e}")
        except Exception as e:
            st.error(f"Error leyendo el archivo: {e}")


# ====================================================================
# ====================================================================
# TAB 3: GESTIÓN DE PREDIOS (COMPLETO Y CORREGIDO)
# ====================================================================
with tab_predios:
    st.header("🏡 Gestión de Predios (Desde GeoJSON)")
    
    # 1. DEFINICIÓN DE SUB-PESTAÑAS (¡Esto es lo que faltaba o estaba mal ubicado!)
    sub_edit_p, sub_crear_p, sub_carga_p = st.tabs(["✏️ Editar Predio", "➕ Crear Predio", "📥 Carga GeoJSON"])

    # ----------------------------------------------------------------
    # SUB-TAB 1: EDITAR PREDIO
    # ----------------------------------------------------------------
    with sub_edit_p:
        engine = get_engine()
        if engine:
            try:
                with engine.connect() as conn:
                    # Buscador ligero
                    df_lista = pd.read_sql(text("SELECT id_predio, nombre_predio FROM predios ORDER BY nombre_predio"), conn)
                    if not df_lista.empty:
                        df_lista['display'] = df_lista['nombre_predio'] + " (" + df_lista['id_predio'].astype(str) + ")"
                        sel_predio = st.selectbox("🔍 Buscar Predio:", df_lista['display'].tolist(), index=None, placeholder="Escribe el nombre de la finca...")
                        
                        if sel_predio:
                            id_p = sel_predio.split('(')[-1].replace(')', '')
                            df_full = pd.read_sql(text("SELECT * FROM predios WHERE id_predio = :id"), conn, params={"id": id_p})
                            
                            if not df_full.empty:
                                data = df_full.iloc[0]
                                st.divider()
                                with st.form("form_edit_predio"):
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        n_nom = st.text_input("Nombre Predio", value=data['nombre_predio'])
                                        n_prop = st.text_input("Propietario", value=data['propietario'] if data['propietario'] else "")
                                        n_ver = st.text_input("Vereda", value=data['vereda'] if data['vereda'] else "")
                                    with c2:
                                        n_mun = st.text_input("Municipio", value=data['municipio'] if data['municipio'] else "")
                                        n_area = st.number_input("Área (Hectáreas)", value=float(data['area_ha']) if data['area_ha'] else 0.0)
                                        st.text_input("ID (No editable)", value=data['id_predio'], disabled=True)
                                    
                                    if st.form_submit_button("💾 Actualizar Predio"):
                                        conn.execute(text("""
                                            UPDATE predios SET nombre_predio=:n, propietario=:p, vereda=:v, 
                                            municipio=:m, area_ha=:a WHERE id_predio=:id
                                        """), {"n": n_nom, "p": n_prop, "v": n_ver, "m": n_mun, "a": n_area, "id": id_p})
                                        conn.commit()
                                        st.success("✅ Predio actualizado.")
                                        st.rerun()
                    else:
                        st.info("No hay predios registrados aún.")
            except Exception as e:
                st.error(f"Error: {e}")

    # ----------------------------------------------------------------
    # SUB-TAB 2: CREAR PREDIO
    # ----------------------------------------------------------------
    with sub_crear_p:
        with st.form("form_create_predio"):
            c1, c2 = st.columns(2)
            with c1:
                new_id = st.text_input("ID Predio (Único)", placeholder="Ej: PRE-001")
                new_nom = st.text_input("Nombre Finca")
                new_prop = st.text_input("Nombre Propietario")
            with c2:
                new_mun = st.text_input("Municipio")
                new_ver = st.text_input("Vereda")
                new_area = st.number_input("Área (ha)", min_value=0.0)
            
            if st.form_submit_button("🚀 Registrar Predio"):
                if new_id and new_nom:
                    engine = get_engine()
                    with engine.connect() as conn:
                        try:
                            conn.execute(text("""
                                INSERT INTO predios (id_predio, nombre_predio, propietario, municipio, vereda, area_ha)
                                VALUES (:id, :nom, :prop, :mun, :ver, :area)
                            """), {"id": new_id, "nom": new_nom, "prop": new_prop, "mun": new_mun, "ver": new_ver, "area": new_area})
                            conn.commit()
                            st.success("Predio creado exitosamente.")
                        except Exception as e:
                            st.error(f"Error: {e}")

    # ----------------------------------------------------------------
    # SUB-TAB 3: CARGA MASIVA GEOJSON (PREDIOS)
    # ----------------------------------------------------------------
    with sub_carga_p:
        st.info("Sube el archivo **PrediosEjecutados.geojson**. Se usarán los campos: PK_PREDIOS, NOMBRE_PRE, NOMB_MPIO, AREA_HA.")
        
        up_geo = st.file_uploader("Arrastra 'PrediosEjecutados.geojson'", type=["geojson", "json"], key="up_predios_json")
        
        if up_geo:
            if st.button("🚀 Procesar Predios"):
                try:
                    data = json.load(up_geo)
                    
                    if "features" not in data:
                        st.error("❌ El archivo no tiene el formato GeoJSON correcto.")
                    else:
                        rows = []
                        with st.spinner(f"Procesando {len(data['features'])} predios..."):
                            for feature in data['features']:
                                props = feature.get("properties", {})
                                geom = feature.get("geometry", {})
                                
                                # A. Calcular Centroide (Lat/Lon)
                                lat, lon = 0.0, 0.0
                                try:
                                    if geom:
                                        if geom.get('type') == 'Point':
                                            lon, lat = geom['coordinates']
                                        elif geom.get('type') in ['Polygon', 'MultiPolygon']:
                                            coords_raw = geom['coordinates']
                                            # Función auxiliar para aplanar coordenadas
                                            def flatten_coords(c):
                                                if len(c) > 0 and isinstance(c[0], (float, int)): return [c]
                                                out = []
                                                for i in c: out.extend(flatten_coords(i))
                                                return out
                                            
                                            all_points = flatten_coords(coords_raw)
                                            df_c = pd.DataFrame(all_points, columns=['lon', 'lat'])
                                            lat = df_c['lat'].mean()
                                            lon = df_c['lon'].mean()
                                except:
                                    pass 

                                # B. Mapeo EXACTO según tus datos
                                rows.append({
                                    "id_predio": str(props.get('PK_PREDIOS', 'SIN_ID')),
                                    "nombre_predio": props.get('NOMBRE_PRE', 'Sin Nombre'),
                                    "propietario": props.get('PROPIETARIO', 'Desconocido'),
                                    "municipio": props.get('NOMB_MPIO', ''),
                                    "vereda": props.get('NOMBRE_VER', ''),
                                    "area_ha": float(props.get('AREA_HA', 0.0)),
                                    "latitud": lat,
                                    "longitud": lon
                                })

                        # 2. Subir a Base de Datos
                        df_upload = pd.DataFrame(rows).drop_duplicates(subset=['id_predio'])
                        st.write(f"✅ Se detectaron {len(df_upload)} predios únicos.", df_upload.head(3))
                        
                        engine = get_engine()
                        with engine.connect() as conn:
                            count = 0
                            for _, row in df_upload.iterrows():
                                upsert_q = text("""
                                    INSERT INTO predios (id_predio, nombre_predio, propietario, municipio, vereda, area_ha, latitud, longitud)
                                    VALUES (:id, :nom, :prop, :mun, :ver, :area, :lat, :lon)
                                    ON CONFLICT (id_predio) DO UPDATE SET
                                    nombre_predio = EXCLUDED.nombre_predio,
                                    municipio = EXCLUDED.municipio,
                                    vereda = EXCLUDED.vereda,
                                    area_ha = EXCLUDED.area_ha,
                                    latitud = EXCLUDED.latitud,
                                    longitud = EXCLUDED.longitud;
                                """)
                                conn.execute(upsert_q, {
                                    "id": row['id_predio'], "nom": row['nombre_predio'], "prop": row['propietario'],
                                    "mun": row['municipio'], "ver": row['vereda'], "area": row['area_ha'],
                                    "lat": row['latitud'], "lon": row['longitud']
                                })
                                count += 1
                            conn.commit()
                            
                        st.success(f"✅ ¡Éxito! Base de datos actualizada con {count} predios.")
                        st.balloons()

                except Exception as e:
                    st.error(f"Error procesando: {e}")


# ====================================================================
# TAB 4: GESTIÓN DE CUENCAS (COMPLETO Y CORREGIDO)
# ====================================================================
with tab_cuencas:
    st.header("🌊 Gestión de Cuencas (Desde GeoJSON)")
    
    # 1. DEFINICIÓN DE SUB-PESTAÑAS (¡Esto es lo que faltaba!)
    sub_edit_c, sub_crear_c, sub_carga_c = st.tabs(["✏️ Editar Cuenca", "➕ Registrar Cuenca", "📥 Carga GeoJSON"])

    # ----------------------------------------------------------------
    # SUB-TAB 1: EDITAR CUENCA
    # ----------------------------------------------------------------
    with sub_edit_c:
        engine = get_engine()
        if engine:
            try:
                with engine.connect() as conn:
                    df_lista = pd.read_sql(text("SELECT id_cuenca, nombre_cuenca FROM cuencas ORDER BY nombre_cuenca"), conn)
                    if not df_lista.empty:
                        df_lista['display'] = df_lista['nombre_cuenca']
                        sel_cuenca = st.selectbox("🔍 Buscar Cuenca:", df_lista['display'].tolist(), index=None)
                        
                        if sel_cuenca:
                            # Obtener ID basado en nombre (simple)
                            id_c = df_lista[df_lista['display'] == sel_cuenca]['id_cuenca'].values[0]
                            data = pd.read_sql(text("SELECT * FROM cuencas WHERE id_cuenca = :id"), conn, params={"id": id_c}).iloc[0]
                            
                            st.divider()
                            with st.form("form_edit_cuenca"):
                                c1, c2 = st.columns(2)
                                with c1:
                                    n_nom = st.text_input("Nombre Cuenca", value=data['nombre_cuenca'])
                                    n_rio = st.text_input("Río Principal", value=data['rio_principal'] if data['rio_principal'] else "")
                                with c2:
                                    n_area = st.number_input("Área (km2)", value=float(data['area_km2']) if data['area_km2'] else 0.0)
                                    n_mun = st.text_area("Municipios de Influencia", value=data['municipios_influencia'] if data['municipios_influencia'] else "")
                                
                                if st.form_submit_button("💾 Guardar Cambios"):
                                    conn.execute(text("""
                                        UPDATE cuencas SET nombre_cuenca=:n, rio_principal=:r, area_km2=:a, municipios_influencia=:m
                                        WHERE id_cuenca=:id
                                    """), {"n": n_nom, "r": n_rio, "a": n_area, "m": n_mun, "id": id_c})
                                    conn.commit()
                                    st.success("✅ Cuenca actualizada.")
                                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # ----------------------------------------------------------------
    # SUB-TAB 2: CREAR CUENCA
    # ----------------------------------------------------------------
    with sub_crear_c:
        with st.form("form_new_cuenca"):
            id_new = st.text_input("ID Cuenca (Ej: 2701-02)")
            nom_new = st.text_input("Nombre Cuenca")
            area_new = st.number_input("Área (km2)")
            
            if st.form_submit_button("🚀 Crear Cuenca"):
                if id_new and nom_new:
                    engine = get_engine()
                    with engine.connect() as conn:
                        try:
                            conn.execute(text("INSERT INTO cuencas (id_cuenca, nombre_cuenca, area_km2) VALUES (:id, :n, :a)"),
                                         {"id": id_new, "n": nom_new, "a": area_new})
                            conn.commit()
                            st.success("Cuenca registrada.")
                        except Exception as e:
                            st.error(f"Error: {e}")

    # ----------------------------------------------------------------
    # SUB-TAB 3: CARGA MASIVA GEOJSON (CUENCAS)
    # ----------------------------------------------------------------
    with sub_carga_c:
        st.info("Sube el archivo **SubcuencasAinfluencia.geojson**. Se usarán los campos: COD, SUBC_LBL, Shape_Area, SZH.")
        
        up_cuenca = st.file_uploader("Arrastra 'SubcuencasAinfluencia.geojson'", type=["geojson", "json"], key="up_cuencas_json")
        
        if up_cuenca and st.button("🚀 Procesar Cuencas"):
            try:
                data = json.load(up_cuenca)
                rows = []
                
                with st.spinner(f"Procesando {len(data['features'])} cuencas..."):
                    for feature in data['features']:
                        props = feature.get("properties", {})
                        
                        # Conversión de Area (m2 a km2)
                        area_m2 = float(props.get('Shape_Area', 0.0))
                        area_km2 = area_m2 / 1_000_000  # 1 km2 = 1,000,000 m2
                        
                        rows.append({
                            # Usamos 'COD' como ID principal (ej: 2701-02-20-50)
                            "id_cuenca": str(props.get('COD', props.get('OBJECTID', 'SIN_ID'))),
                            # Usamos 'SUBC_LBL' como nombre (ej: R. Chico)
                            "nombre_cuenca": props.get('SUBC_LBL', props.get('N_NSS1', 'Sin Nombre')),
                            "area_km2": area_km2,
                            # Usamos 'SZH' (Subzona Hidrográfica) como río/sistema principal
                            "rio_principal": props.get('SZH', ''),
                            # Usamos 'Zona' o 'depto_region' para ubicación
                            "municipios_influencia": f"{props.get('Zona', '')} - {props.get('depto_region', '')}".strip()
                        })
                
                df_cuencas = pd.DataFrame(rows).drop_duplicates(subset=['id_cuenca'])
                st.write(f"✅ Se detectaron {len(df_cuencas)} cuencas. Ejemplo:", df_cuencas.head(3))
                
                engine = get_engine()
                with engine.connect() as conn:
                    count = 0
                    for _, row in df_cuencas.iterrows():
                        q = text("""
                            INSERT INTO cuencas (id_cuenca, nombre_cuenca, area_km2, rio_principal, municipios_influencia)
                            VALUES (:id, :nom, :area, :rio, :mun)
                            ON CONFLICT (id_cuenca) DO UPDATE SET
                            nombre_cuenca = EXCLUDED.nombre_cuenca,
                            area_km2 = EXCLUDED.area_km2,
                            rio_principal = EXCLUDED.rio_principal;
                        """)
                        conn.execute(q, {
                            "id": row['id_cuenca'], "nom": row['nombre_cuenca'], 
                            "area": row['area_km2'], "rio": row['rio_principal'], "mun": row['municipios_influencia']
                        })
                        count += 1
                    conn.commit()
                
                st.success(f"✅ ¡Éxito! Se cargaron/actualizaron {count} cuencas.")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error procesando Cuencas: {e}")

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