import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import os
import io
import base64
import matplotlib.pyplot as plt

# --- CONSTANTES Y LEYENDAS ---
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

def process_land_cover_raster(raster_path, gdf_mask=None, scale_factor=10):
    """
    Lee el raster. Si hay gdf_mask, recorta. Si no, lee regional con downscale.
    """
    if not os.path.exists(raster_path):
        return None, None, None, None

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        crs = src.crs

        if gdf_mask is not None:
            # --- MODO RECORTE (CUENCA) ---
            if gdf_mask.crs != src.crs:
                gdf_proj = gdf_mask.to_crs(src.crs)
            else:
                gdf_proj = gdf_mask
            
            try:
                out_image, out_transform = mask(src, gdf_proj.geometry, crop=True)
                data = out_image[0]
            except ValueError:
                return None, None, None, None
        else:
            # --- MODO REGIONAL (OPTIMIZADO) ---
            new_height = int(src.height / scale_factor)
            new_width = int(src.width / scale_factor)
            
            data = src.read(
                1,
                out_shape=(new_height, new_width),
                resampling=Resampling.nearest
            )
            out_transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            )

    return data, out_transform, crs, nodata


def calculate_land_cover_stats(data, transform, nodata, manual_area_km2=None):
    """
    Calcula estadísticas. Usa 'manual_area_km2' para corregir el volumen total.
    """
    valid_pixels = data[(data != nodata) & (data > 0)]
    
    if valid_pixels.size == 0:
        return pd.DataFrame(), 0

    # Área de pixel en km2
    pixel_area_km2 = (abs(transform[0] * transform[4])) / 1e6

    unique, counts = np.unique(valid_pixels, return_counts=True)
    
    # Cálculo total basado en pixeles
    calc_total_area = counts.sum() * pixel_area_km2
    
    # AJUSTE DE VOLUMEN
    if manual_area_km2 and manual_area_km2 > 0:
        factor = manual_area_km2 / calc_total_area if calc_total_area > 0 else 1
        final_total_area = manual_area_km2
    else:
        factor = 1
        final_total_area = calc_total_area
    
    rows = []
    for val, count in zip(unique, counts):
        area = (count * pixel_area_km2) * factor
        if final_total_area > 0:
            pct = (area / final_total_area) * 100
        else:
            pct = 0
        
        rows.append({
            "Cobertura": LAND_COVER_LEGEND.get(val, f"Clase {val}"),
            "Área (km²)": area,
            "%": pct,
            "Color": LAND_COVER_COLORS.get(val, "#808080")
        })
    
    df = pd.DataFrame(rows).sort_values("%", ascending=False)
    return df, final_total_area


def get_raster_img_b64(data, nodata):
    """
    Convierte la matriz a imagen PNG Base64.
    """
    # Crear matriz RGBA
    rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
    
    # Colorear
    for val, hex_c in LAND_COVER_COLORS.items():
        if isinstance(hex_c, str):
            hex_c = hex_c.lstrip('#')
            r, g, b = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
            
            mask_val = (data == val)
            rgba[mask_val, 0] = r
            rgba[mask_val, 1] = g
            rgba[mask_val, 2] = b
            rgba[mask_val, 3] = 180 # Opacidad
    
    # Transparencia total para NoData
    rgba[(data == 0) | (data == nodata), 3] = 0
    
    # Guardar en memoria como PNG
    image_data = io.BytesIO()
    plt.imsave(image_data, rgba, format='png')
    image_data.seek(0)
    b64_encoded = base64.b64encode(image_data.read()).decode('utf-8')
    
    return f"data:image/png;base64,{b64_encoded}"


def get_land_cover_at_point(lat, lon, raster_path):
    """Obtiene la cobertura en un punto."""
    if not os.path.exists(raster_path):
        return "Raster no encontrado"
    try:
        with rasterio.open(raster_path) as src:
            val_gen = src.sample([(lon, lat)])
            val = next(val_gen)[0]
            if val == src.nodata or val == 0:
                return "Sin Datos"
            return LAND_COVER_LEGEND.get(int(val), f"Clase {val}")
    except Exception:
        return "Error Raster"


def calculate_scs_runoff(cn, ppt_mm):
    if cn >= 100:
        return ppt_mm
    if cn <= 0:
        return 0
    
    s = (25400 / cn) - 254
    ia = 0.2 * s
    
    if ppt_mm > ia:
        return ((ppt_mm - ia) ** 2) / (ppt_mm - ia + s)
    else:
        return 0


def calculate_weighted_cn(df_stats, cn_config):
    cn_pond = 0
    total_pct = 0
    for _, row in df_stats.iterrows():
        cob = row["Cobertura"]
        pct = row["%"]
        val = 85 # Default
        if "Bosque" in cob:
            val = cn_config['bosque']
        elif "Pasto" in cob or "Herbácea" in cob:
            val = cn_config['pasto']
        elif "Urban" in cob:
            val = cn_config['urbano']
        elif "Agua" in cob:
            val = 100
        elif "Suelo" in cob or "Degradada" in cob:
            val = cn_config['suelo']
        elif "Cultivo" in cob or "Agrícola" in cob:
            val = cn_config['cultivo']
        
        cn_pond += val * pct / 100
        total_pct += pct
    
    if total_pct > 0:
        return (cn_pond / total_pct) * 100
    else:
        return 0