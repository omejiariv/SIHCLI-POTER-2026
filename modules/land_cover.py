# modules/land_cover.py

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import os

# --- CONSTANTES Y DICCIONARIOS ---

LAND_COVER_LEGEND = {
    1: "Zonas Urbanas", 2: "Cultivos Transitorios", 3: "Pastos", 4: "Áreas Agrícolas",
    5: "Bosques", 6: "Vegetación Herbácea", 7: "Áreas Abiertas", 8: "Aguas",
    9: "Bosque Fragmentado", 10: "Vegetación Secundaria", 11: "Zonas Degradadas",
    12: "Humedales", 13: "Otros / Sin Clasificar"
}

LAND_COVER_COLORS = {
    1: "#A9A9A9", 2: "#FFFF00", 3: "#FFA500", 4: "#FFD700",
    5: "#006400", 6: "#32CD32", 7: "#F4A460", 8: "#0000FF",
    9: "#228B22", 10: "#9ACD32", 11: "#8B4513", 12: "#00CED1", 13: "#FF00FF"
}

# --- FUNCIONES DE PROCESAMIENTO ---

def process_land_cover_raster(raster_path, gdf_mask=None, scale_factor=15):
    """
    Lee el raster de coberturas. 
    - Si gdf_mask (geometría) es None, lee en modo regional (con downscale).
    - Si gdf_mask existe, recorta el raster a la cuenca.
    
    Retorna:
    - data (numpy array): Matriz de datos.
    - transform (Affine): Transformación geoespacial.
    - crs: Sistema de coordenadas.
    - nodata: Valor de no-data.
    """
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"El archivo raster no existe en: {raster_path}")

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        crs = src.crs

        if gdf_mask is not None:
            # --- MODO RECORTE (CUENCA) ---
            # Asegurar proyección
            if gdf_mask.crs != src.crs:
                gdf_proj = gdf_mask.to_crs(src.crs)
            else:
                gdf_proj = gdf_mask
            
            out_image, out_transform = mask(src, gdf_proj.geometry, crop=True)
            data = out_image[0]
        else:
            # --- MODO REGIONAL (COMPLETO / DOWNSCALE) ---
            # Leemos con factor de escala para rendimiento visual
            new_height = int(src.height / scale_factor)
            new_width = int(src.width / scale_factor)
            
            data = src.read(
                1,
                out_shape=(new_height, new_width),
                resampling=Resampling.nearest
            )
            # Ajustar transformación
            out_transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            )

    return data, out_transform, crs, nodata


def calculate_land_cover_stats(data, transform, nodata):
    """
    Calcula las estadísticas de área por cobertura.
    Retorna un DataFrame y el área total en km2.
    """
    # Filtrar datos válidos
    valid_pixels = data[(data != nodata) & (data > 0)]
    
    if valid_pixels.size == 0:
        return pd.DataFrame(), 0

    # Calcular área de un pixel en km2
    # transform[0] es ancho pixel, transform[4] es alto (negativo)
    pixel_area_km2 = (transform[0] * -transform[4]) / 1e6

    unique, counts = np.unique(valid_pixels, return_counts=True)
    total_area_km2 = counts.sum() * pixel_area_km2
    
    rows = []
    for val, count in zip(unique, counts):
        area = count * pixel_area_km2
        pct = (area / total_area_km2) * 100
        
        rows.append({
            "Cobertura": LAND_COVER_LEGEND.get(val, f"Clase {val}"),
            "Área (km²)": area,
            "%": pct,
            "Color": LAND_COVER_COLORS.get(val, "#808080")
        })
    
    df = pd.DataFrame(rows).sort_values("%", ascending=False)
    return df, total_area_km2


def calculate_scs_runoff(cn, ppt_mm):
    """
    Calcula la escorrentía (Q) usando el método SCS-CN.
    """
    if cn >= 100: return ppt_mm
    if cn <= 0: return 0
    
    s = (25400 / cn) - 254
    ia = 0.2 * s
    
    if ppt_mm > ia:
        q = ((ppt_mm - ia) ** 2) / (ppt_mm - ia + s)
        return q
    else:
        return 0

def calculate_weighted_cn(df_stats, cn_config):
    """
    Calcula el CN ponderado actual basado en el DataFrame de estadísticas.
    cn_config: diccionario con claves 'bosque', 'pasto', 'cultivo', 'urbano', 'suelo'.
    """
    cn_ponderado = 0
    total_pct = 0
    
    for _, row in df_stats.iterrows():
        cob = row["Cobertura"]
        pct = row["%"]
        
        # Lógica simple de asignación (se puede mejorar con mapeo directo)
        if "Bosque" in cob: val = cn_config['bosque']
        elif "Pasto" in cob or "Herbácea" in cob: val = cn_config['pasto']
        elif "Urban" in cob: val = cn_config['urbano']
        elif "Agua" in cob: val = 100
        elif "Suelo" in cob or "Degradada" in cob: val = cn_config['suelo']
        else: val = cn_config['cultivo'] # Default a cultivo/agrícola
        
        cn_ponderado += val * pct / 100
        total_pct += pct
        
    return cn_ponderado