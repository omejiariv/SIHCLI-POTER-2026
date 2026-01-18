# generar_parquet.py

import pandas as pd
import os

# Configuración de archivos
ARCHIVO_CSV_ENTRADA = "data/DatosPptnmes_ENSO.csv"
ARCHIVO_PARQUET_SALIDA = "data/datos_precipitacion_largos.parquet"

def procesar_csv_a_parquet():
    print(f"Leyendo archivo CSV: {ARCHIVO_CSV_ENTRADA}...")
    
    # 1. Leer el CSV original
    # Usamos sep=";" y encoding='latin1' igual que en tu script original
    try:
        df = pd.read_csv(ARCHIVO_CSV_ENTRADA, sep=";", encoding='latin1')
    except FileNotFoundError:
        print(f"ERROR: No se encuentra el archivo {ARCHIVO_CSV_ENTRADA}")
        return

    # 2. Limpieza de nombres de columnas (eliminar espacios y minúsculas)
    df.columns = [col.strip().lower() for col in df.columns]
    
    print("Transformando datos de formato ancho a largo...")

    # 3. Identificar columnas que NO son estaciones (Metadatos e Índices)
    # Basado en tu script, estas son las columnas que NO contienen lluvia directa de estaciones
    columnas_indices = [
        'fecha_mes_año', 'anomalia_oni', 'soi', 'iod', 
        'año', 'mes', 'enso_año', 'enso_mes', 'temp_sst', 'temp_media'
    ]
    
    # Filtramos para quedarnos solo con las que existen realmente en el archivo
    cols_id = [c for c in columnas_indices if c in df.columns]
    
    # Las columnas restantes se asumen como Estaciones (IDs)
    cols_estaciones = [c for c in df.columns if c not in cols_id]
    
    # 4. "Derretir" (Melt) el DataFrame
    # Esto convierte la matriz de muchas columnas (estaciones) en solo 3 columnas: Fecha, Estación, Valor
    df_long = df.melt(
        id_vars=['fecha_mes_año'],      # Columna que se mantiene fija (Fecha)
        value_vars=cols_estaciones,     # Columnas a transformar (Estaciones)
        var_name='id_estacion',         # Nueva columna con el nombre de la cabecera (ID Estación)
        value_name='precipitacion_mm'   # Nueva columna con el valor de la celda
    )

    # 5. Limpieza final de datos
    # Convertir precipitación a numérico (reemplazando comas por puntos si es necesario)
    df_long['precipitacion_mm'] = (
        df_long['precipitacion_mm']
        .astype(str)
        .str.replace(',', '.', regex=False)
    )
    df_long['precipitacion_mm'] = pd.to_numeric(df_long['precipitacion_mm'], errors='coerce')
    
    # Eliminar filas donde no hay fecha o dato de precipitación (opcional, para reducir tamaño)
    df_long = df_long.dropna(subset=['fecha_mes_año', 'precipitacion_mm'])

    print(f"Generando archivo Parquet con {len(df_long)} registros...")

    # 6. Guardar como Parquet
    # Requiere tener instalado pyarrow o fastparquet (pip install pyarrow)
    df_long.to_parquet(ARCHIVO_PARQUET_SALIDA, index=False)
    
    print(f"¡Éxito! Archivo guardado en: {ARCHIVO_PARQUET_SALIDA}")

if __name__ == "__main__":
    procesar_csv_a_parquet()