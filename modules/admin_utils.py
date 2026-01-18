# modules/admin_utils.py

import pandas as pd
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