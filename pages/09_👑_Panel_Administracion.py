# pages/09_👑_Panel_Administracion.py

import streamlit as st
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
import time
from modules import admin_utils  # <--- IMPORTAMOS TU NUEVO MOTOR

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

# ==============================================================================
# TAB 2: GESTIÓN DE PREDIOS (TU CÓDIGO ORIGINAL CONSERVADO)
# ==============================================================================
with tab_predios:
    st.subheader("🏡 Gestión de Predios")
    st.markdown("#### 🔍 Buscar y Editar")
    try:
        engine = get_engine()
        search_term = st.text_input("Buscar (Código, Nombre Finca o Propietario):", placeholder="Ej: La Esperanza...")
        
        if search_term:
            query_search = text("""
                SELECT id, codigo_catastral, nombre_predio, propietario, vereda, municipio, estado_gestion, area_ha 
                FROM predios_gestion 
                WHERE codigo_catastral ILIKE :s OR nombre_predio ILIKE :s OR propietario ILIKE :s
                LIMIT 10
            """)
            with engine.connect() as conn:
                results = pd.read_sql(query_search, conn, params={"s": f"%{search_term.strip()}%"})
            
            if not results.empty:
                opt_p = {f"{r['nombre_predio']} - {r['propietario']} ({r['codigo_catastral']})": r['id'] for _, r in results.iterrows()}
                sel_p_id = st.selectbox("Resultados:", list(opt_p.keys()))
                
                if sel_p_id:
                    curr_p = results[results['id'] == opt_p[sel_p_id]].iloc[0]
                    with st.form("upd_predio"):
                        c1, c2 = st.columns(2)
                        new_nom = c1.text_input("Nombre Finca:", value=curr_p['nombre_predio'])
                        new_prop = c2.text_input("Propietario:", value=curr_p['propietario'])
                        c3, c4 = st.columns(2)
                        new_ver = c3.text_input("Vereda:", value=curr_p['vereda'])
                        new_mun = c4.text_input("Municipio:", value=curr_p['municipio'])
                        new_st = st.selectbox("Estado:", ["Identificado", "En Negociación", "Ejecutado / Conservado", "Descartado"], index=0)
                        
                        if st.form_submit_button("💾 Guardar Cambios"):
                            with engine.connect() as conn:
                                sql = text("UPDATE predios_gestion SET nombre_predio=:n, propietario=:p, vereda=:v, municipio=:m, estado_gestion=:s WHERE id=:i")
                                conn.execute(sql, {"n": new_nom, "p": new_prop, "v": new_ver, "m": new_mun, "s": new_st, "i": curr_p['id']})
                                conn.commit()
                            st.success("✅ Actualizado.")
                            time.sleep(1); st.rerun()
            else: st.warning("No encontrado.")
            
    except Exception as e: st.error(f"Error: {e}")
    
    with st.expander("☁️ Re-Sincronizar Predios (GitHub)", expanded=False):
        if st.button("🔄 Sincronizar Predios", type="primary"):
            with st.status("Procesando...", expanded=True):
                try:
                    url = "https://raw.githubusercontent.com/omejiariv/SIHCLI-POTER-2026/main/data/PrediosEjecutados.geojson"
                    gdf = gpd.read_file(url)
                    if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                    with engine.connect() as conn:
                        conn.execute(text("TRUNCATE TABLE predios_gestion RESTART IDENTITY"))
                        for idx, row in gdf.iterrows():
                            cod = str(row.get('PK_PREDIOS') or "")
                            nom_pre = row.get('NOMBRE_PRE') or 'Sin Nombre'
                            vereda = row.get('NOMBRE_VER') or 'Sin Vereda'
                            muni = row.get('NOMB_MPIO') or row.get('NOM_MPIO') or 'Sin Municipio'
                            area = row.get('AREA_HA') or row.get('Shape_Area') or 0.0
                            sql = text("INSERT INTO predios_gestion (codigo_catastral, nombre_predio, propietario, vereda, municipio, estado_gestion, area_ha, geom) VALUES (:c, :np, 'Por Asignar', :v, :m, 'Identificado', :a, ST_Multi(ST_GeomFromText(:g, 4326)))")
                            conn.execute(sql, {"c": cod, "np": nom_pre, "v": vereda, "m": muni, "a": float(area), "g": row.geometry.wkt})
                        conn.commit()
                    st.success("✅ Sincronización completa.")
                except Exception as e: st.error(f"Error: {e}")

# ==============================================================================
# TAB 3: CUENCAS (TU CÓDIGO ORIGINAL CONSERVADO)
# ==============================================================================
with tab_cuencas:
    st.subheader("🌊 Gestión de Cuencas")
    try:
        engine = get_engine()
        search_c = st.text_input("Buscar Cuenca:", placeholder="Ej: Rio Grande...")
        if search_c:
            q_c = text("SELECT * FROM cuencas_gestion WHERE nombre ILIKE :s LIMIT 10")
            with engine.connect() as conn:
                res_c = pd.read_sql(q_c, conn, params={"s": f"%{search_c}%"})
            if not res_c.empty:
                opt_c = {f"{r['nombre']} ({r['codigo']})": r['id'] for _, r in res_c.iterrows()}
                sel_c_id = st.selectbox("Cuencas:", list(opt_c.keys()))
                if sel_c_id:
                    curr_c = res_c[res_c['id'] == opt_c[sel_c_id]].iloc[0]
                    with st.form("upd_cuenca"):
                        new_name_c = st.text_input("Nombre:", value=curr_c['nombre'])
                        new_obs = st.text_area("Observación:", value=curr_c.get('observacion', ''))
                        if st.form_submit_button("💾 Guardar"):
                            with engine.connect() as conn:
                                conn.execute(text("UPDATE cuencas_gestion SET nombre=:n, observacion=:o WHERE id=:i"), {"n": new_name_c, "o": new_obs, "i": curr_c['id']})
                                conn.commit()
                            st.success("Actualizado."); time.sleep(1); st.rerun()
    except Exception as e: st.error(f"Error: {e}")

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