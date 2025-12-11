import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
import os
import io
import base64
import matplotlib.pyplot as plt

# --- 1. LEYENDA Y COLORES CORREGIDOS (Tus requerimientos) ---
LAND_COVER_LEGEND = {
    1: "Zonas Urbanas", 
    2: "Cultivos Transitorios", 
    3: "Pastos", 
    4: "Áreas Agrícolas Heterogéneas",
    5: "Bosque Denso / Natural", 
    6: "Vegetación Herbácea / Arbustiva", 
    7: "Áreas Abiertas", 
    8: "Bosques Plantados",       # <--- CORREGIDO
    9: "Bosque Fragmentado", 
    10: "Vegetación Secundaria", 
    11: "Zonas Degradadas",
    12: "Humedales", 
    13: "Agua / Cuerpos de Agua"  # <--- CORREGIDO
}

LAND_COVER_COLORS = {
    1: "#A9A9A9",   # Gris
    2: "#FFFF00",   # Amarillo
    3: "#FFA500",   # Naranja
    4: "#FFD700",   # Oro
    5: "#006400",   # Verde Oscuro
    6: "#32CD32",   # Verde Lima
    7: "#F4A460",   # Arena
    8: "#2E8B57",   # Verde Mar (Plantados)
    9: "#228B22",   # Verde Forestal
    10: "#9ACD32",  # Verde Claro
    11: "#8B4513",  # Café
    12: "#00CED1",  # Turquesa
    13: "#0000FF"   # Azul Puro (Agua)
}

# --- 2. PROCESAMIENTO ---

def process_land_cover_raster(raster_path, gdf_mask=None, scale_factor=10):
    """Lee el raster. Si hay cuenca (gdf_mask), recorta exactamente."""
    if not os.path.exists(raster_path):
        return None, None, None, None

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        crs = src.crs

        if gdf_mask is not None:
            # MODO CUENCA
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
            # MODO REGIONAL (Optimizado)
            new_height = int(src.height / scale_factor)
            new_width = int(src.width / scale_factor)
            data = src.read(1, out_shape=(new_height, new_width), resampling=Resampling.nearest)
            out_transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]), (src.height / data.shape[-2])
            )

    return data, out_transform, crs, nodata

def calculate_land_cover_stats(data, transform, nodata, manual_area_km2=None):
    """Calcula áreas y porcentajes. Corrige el total si se da el área vectorial."""
    valid_pixels = data[(data != nodata) & (data > 0)]
    if valid_pixels.size == 0: return pd.DataFrame(), 0

    pixel_area_km2 = (abs(transform[0] * transform[4])) / 1e6
    unique, counts = np.unique(valid_pixels, return_counts=True)
    
    calc_total_area = counts.sum() * pixel_area_km2
    
    # AJUSTE PARA VOLUMEN CORRECTO
    final_total_area = calc_total_area
    factor = 1.0
    if manual_area_km2 and manual_area_km2 > 0:
        final_total_area = manual_area_km2
        if calc_total_area > 0:
            factor = manual_area_km2 / calc_total_area
    
    rows = []
    for val, count in zip(unique, counts):
        area = (count * pixel_area_km2) * factor
        pct = (area / final_total_area) * 100
        rows.append({
            "ID": val,
            "Cobertura": LAND_COVER_LEGEND.get(val, f"Clase {val}"),
            "Área (km²)": area,
            "%": pct,
            "Color": LAND_COVER_COLORS.get(val, "#808080")
        })
    
    return pd.DataFrame(rows).sort_values("%", ascending=False), final_total_area

def get_raster_img_b64(data, nodata):
    """Genera la imagen para el mapa (Estable)."""
    rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
    for val, hex_c in LAND_COVER_COLORS.items():
        if isinstance(hex_c, str):
            hex_c = hex_c.lstrip('#')
            r, g, b = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
            
            mask_val = (data == val)
            rgba[mask_val, 0] = r
            rgba[mask_val, 1] = g
            rgba[mask_val, 2] = b
            rgba[mask_val, 3] = 180 # Opacidad
            
    rgba[(data == 0) | (data == nodata), 3] = 0
    
    image_data = io.BytesIO()
    plt.imsave(image_data, rgba, format='png')
    image_data.seek(0)
    return f"data:image/png;base64,{base64.b64encode(image_data.read()).decode('utf-8')}"

# --- 3. FUNCIONES NUEVAS (HOVER Y LEYENDA) ---

def vectorize_raster(data, transform, crs, nodata):
    """Vectoriza para permitir Hover en Folium."""
    mask_arr = (data != nodata) & (data != 0)
    # Límite de seguridad para no colgar el navegador
    if np.count_nonzero(mask_arr) > 100000: return gpd.GeoDataFrame()

    shapes_gen = shapes(data, mask=mask_arr, transform=transform)
    geoms, values = [], []
    for geom, val in shapes_gen:
        geoms.append(shape(geom))
        values.append(val)
        
    if not geoms: return gpd.GeoDataFrame()
    
    gdf = gpd.GeoDataFrame({'ID': values}, geometry=geoms, crs=crs)
    gdf['Cobertura'] = gdf['ID'].map(lambda x: LAND_COVER_LEGEND.get(int(x), f"Clase {int(x)}"))
    gdf['Color'] = gdf['ID'].map(lambda x: LAND_COVER_COLORS.get(int(x), "#808080"))
    return gdf.to_crs(epsg=4326)

def generate_legend_html():
    """Crea la leyenda flotante HTML."""
    html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; 
        background-color: rgba(255, 255, 255, 0.9); padding: 10px; 
        border: 1px solid #ccc; border-radius: 5px; font-family: sans-serif;
        font-size: 11px; max-height: 250px; overflow-y: auto; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);">
        <b style="display:block; margin-bottom:5px;">Leyenda Coberturas</b>
    """
    for id_cov, name in sorted(LAND_COVER_LEGEND.items()):
        color = LAND_COVER_COLORS.get(id_cov, "#808080")
        html += f'<div style="margin-bottom:3px;"><i style="background:{color}; width:12px; height:12px; display:inline-block; margin-right:5px; border:1px solid #999;"></i>{name}</div>'
    html += "</div>"
    return html

def get_tiff_bytes(data, transform, crs, nodata):
    """Prepara descarga TIFF."""
    mem_file = io.BytesIO()
    with rasterio.open(
        mem_file, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype, crs=crs, transform=transform, nodata=nodata
    ) as dst:
        dst.write(data, 1)
    mem_file.seek(0)
    return mem_file

# --- 4. CÁLCULOS SCS ---
def calculate_weighted_cn(df_stats, cn_config):
    cn_pond = 0; total_pct = 0
    for _, row in df_stats.iterrows():
        cob = row["Cobertura"]; pct = row["%"]
        val = 85
        if "Bosque" in cob: val = cn_config['bosque']
        elif "Pasto" in cob or "Herbácea" in cob: val = cn_config['pasto']
        elif "Urban" in cob: val = cn_config['urbano']
        elif "Agua" in cob: val = 100
        elif "Suelo" in cob or "Degradada" in cob: val = cn_config['suelo']
        elif "Cultivo" in cob: val = cn_config['cultivo']
        cn_pond += val * pct / 100
        total_pct += pct
    return (cn_pond / total_pct) * 100 if total_pct > 0 else 0

def calculate_scs_runoff(cn, ppt_mm):
    if cn >= 100: return ppt_mm
    if cn <= 0: return 0
    s = (25400 / cn) - 254; ia = 0.2 * s
    return ((ppt_mm - ia) ** 2) / (ppt_mm - ia + s) if ppt_mm > ia else 0

def get_land_cover_at_point(lat, lon, raster_path):
    if not os.path.exists(raster_path): return "Raster no encontrado"
    try:
        with rasterio.open(raster_path) as src:
            val = next(src.sample([(lon, lat)]))[0]
            if val == src.nodata or val == 0: return "Sin Datos"
            return LAND_COVER_LEGEND.get(int(val), f"Clase {val}")
    except: return "Error Raster"