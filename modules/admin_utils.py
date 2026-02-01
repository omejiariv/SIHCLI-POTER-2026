# modules/admin_utils.py


import os
import streamlit as st
from supabase import create_client, Client
import pandas as pd
import tempfile

from modules.utils import standardize_numeric_column

def parse_spanish_date(date_str):
    """
    Convierte fechas de texto tipo 'feb-80', 'ene-99' a objetos fecha reales.
    Maneja el problema del idioma español y los años de dos dígitos.
    """
    if pd.isna(date_str): return None
    date_str = str(date_str).lower().strip()
    
    # Diccionario de traducción
    meses = {
        'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12
    }
    
    try:
        # Detectar separador (- o /)
        sep = '-' if '-' in date_str else '/'
        parts = date_str.split(sep)
        
        if len(parts) != 2: return None # Formato desconocido
        
        m_str, y_str = parts[0], parts[1]
        
        # Traducir mes
        month = meses.get(m_str[:3]) # Tomamos las primeras 3 letras
        if not month: return None
        
        # Traducir año
        year = int(y_str)
        if year < 100:
            # Lógica de pivote: Si es mayor a 40, es 19xx (ej: 80 -> 1980). Si no, 20xx.
            year = 1900 + year if year > 40 else 2000 + year
            
        # Retornar fecha primer día del mes
        return pd.Timestamp(year=year, month=month, day=1)
        
    except Exception:
        return None

def procesar_archivo_precipitacion(uploaded_file):
    """
    Procesador ETL Principal: Carga, Limpia, Traduce Fechas y Estandariza.
    """
    try:
        # 1. Cargar CSV
        df_raw = pd.read_csv(
            uploaded_file, 
            sep=';', 
            encoding='latin1',
            low_memory=False
        )

        # 2. Definir columnas de METADATOS
        columnas_a_excluir = [
            'fecha_mes_año', 'anomalia_oni', 'soi', 'iod', 'temp_sst', 
            'temp_media', 'id', 'fecha', 'mes', 'año', 'id_estacio', 
            'nom_est', 'unnamed', 'enso_año', 'enso_mes'
        ]

        # 3. Identificar columnas dinámicamente
        columnas_id = [
            col for col in df_raw.columns 
            if any(ex_col in col.lower() for ex_col in columnas_a_excluir)
        ]
        columnas_estaciones = [col for col in df_raw.columns if col not in columnas_id]
        
        if not columnas_estaciones:
            return None, "No se detectaron columnas de estaciones."

        # 4. Transformación (MELT)
        df_long = df_raw.melt(
            id_vars=columnas_id, 
            value_vars=columnas_estaciones, 
            var_name='id_estacion', 
            value_name='precipitation'
        )

        # 5. Limpieza Numérica
        df_long['precipitation'] = standardize_numeric_column(df_long['precipitation'])
        df_long['id_estacion'] = df_long['id_estacion'].astype(str).str.strip()

        # 6. Estandarización y TRADUCCIÓN de Fechas
        col_fecha = next((c for c in columnas_id if 'fecha' in c.lower()), None)
        
        if col_fecha:
            # Renombrar
            df_long = df_long.rename(columns={col_fecha: 'fecha_mes_año'})
            
            # --- AQUÍ OCURRE LA MAGIA ---
            # Aplicamos la función traductora a toda la columna
            df_long['fecha_mes_año'] = df_long['fecha_mes_año'].apply(parse_spanish_date)
            
            # Eliminar fechas que no se pudieron entender (NaT)
            df_long = df_long.dropna(subset=['fecha_mes_año'])
        else:
            return None, "No se encontró columna de fecha."

        # 7. Limpieza final
        df_long = df_long.dropna(subset=['precipitation'])

        return df_long, None

    except Exception as e:
        return None, str(e)

# --- FUNCIONES DE GESTIÓN DE STORAGE (RASTERS) ---

# Inicializar cliente Supabase (Singleton)
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["SUPABASE_URL"]
    key = st.secrets["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

def get_raster_list(bucket_name="rasters"):
    """Lista archivos en el bucket de Supabase."""
    try:
        supabase = init_supabase()
        res = supabase.storage.from_(bucket_name).list()
        return res # Retorna lista de objetos
    except Exception as e:
        print(f"Error listando bucket: {e}")
        return []

def upload_raster_to_storage(file_bytes, file_name, bucket_name="rasters"):
    """Sube un archivo raster al bucket."""
    try:
        supabase = init_supabase()
        # file_options param is crucial for overwriting
        res = supabase.storage.from_(bucket_name).upload(
            path=file_name,
            file=file_bytes,
            file_options={"content-type": "image/tiff", "upsert": "true"}
        )
        return True, f"✅ '{file_name}' subido a la nube exitosamente."
    except Exception as e:
        return False, f"❌ Error subiendo: {str(e)}"

def delete_raster_from_storage(file_name, bucket_name="rasters"):
    """Elimina un archivo del bucket."""
    try:
        supabase = init_supabase()
        res = supabase.storage.from_(bucket_name).remove([file_name])
        return True, f"🗑️ '{file_name}' eliminado."
    except Exception as e:
        return False, f"❌ Error eliminando: {str(e)}"



@st.cache_resource
def init_supabase():
    # Ahora que corregiste la carpeta .streamlit, esto funcionará
    url = st.secrets["supabase"]["SUPABASE_URL"]
    key = st.secrets["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

def get_raster_list(bucket_name="rasters"):
    try:
        supabase = init_supabase()
        res = supabase.storage.from_(bucket_name).list()
        return res
    except Exception as e:
        return []

def upload_raster_to_storage(file_bytes, file_name, bucket_name="rasters"):
    try:
        supabase = init_supabase()
        res = supabase.storage.from_(bucket_name).upload(
            path=file_name,
            file=file_bytes,
            file_options={"content-type": "image/tiff", "upsert": "true"}
        )
        return True, f"✅ Carga exitosa: {file_name}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def delete_raster_from_storage(file_name, bucket_name="rasters"):
    try:
        supabase = init_supabase()
        supabase.storage.from_(bucket_name).remove([file_name])
        return True, f"🗑️ Eliminado: {file_name}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def download_raster_to_temp(file_name, bucket_name="rasters"):
    """
    Descarga un archivo de Supabase y devuelve la ruta temporal local.
    """
    try:
        supabase = init_supabase()
        # Descargamos los bytes
        data = supabase.storage.from_(bucket_name).download(file_name)
        
        # Guardamos en un archivo temporal que el sistema pueda leer
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(data)
            return tmp.name # Retornamos la ruta (ej: /tmp/tmpxyz.tif)
    except Exception as e:
        print(f"Error descargando {file_name}: {e}")
        return None