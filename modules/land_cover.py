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

# --- FUNCIONES CORE ---

def process_land_cover_raster(raster_path, gdf_mask=None, scale_factor=10):
    """Lee y procesa el raster (Regional o Recortado)."""
    if not os.path.exists(raster_path):
        return None, None, None, None

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        crs = src.crs

        if gdf_mask is not None:
            # Asegurar proyección
            if gdf_mask.crs != src.crs:
                gdf_proj = gdf_mask.to_crs(src.crs)
            else:
                gdf_proj = gdf_mask
            
            try:
                # crop=True recorta los bordes al bounding box de la geometría
                out_image, out_transform = mask(src, gdf_proj.geometry, crop=True)
                data = out_image[0]
            except ValueError:
                return None, None, None, None
        else:
            # Modo Regional (Downscale)
            new_height = int(src.height / scale_factor)
            new_width = int(src.width / scale_factor)
            data = src.read(1, out_shape=(new_height, new_width), resampling=Resampling.nearest)
            out_transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            )

    return data, out_transform, crs, nodata

def calculate_land_cover_stats(data, transform, nodata, manual_area_km2=None):
    """Calcula estadísticas y corrige el área total si se provee."""
    valid_pixels = data[(data != nodata) & (data > 0)]
    if valid_pixels.size == 0: return pd.DataFrame(), 0

    pixel_area_km2 = (abs(transform[0] * transform[4])) / 1e6
    unique, counts = np.unique(valid_pixels, return_counts=True)
    
    calc_total_area = counts.sum() * pixel_area_km2
    factor = (manual_area_km2 / calc_total_area) if (manual_area_km2 and manual_area_km2 > 0 and calc_total_area > 0) else 1
    final_total_area = manual_area_km2 if (manual_area_km2 and manual_area_km2 > 0) else calc_total_area
    
    rows = []
    for val, count in zip(unique, counts):
        area = (count * pixel_area_km2) * factor
        pct = (area / final_total_area) * 100 if final_total_area > 0 else 0
        rows.append({
            "ID": val,
            "Cobertura": LAND_COVER_LEGEND.get(val, f"Clase {val}"),
            "Área (km²)": area,
            "%": pct,
            "Color": LAND_COVER_COLORS.get(val, "#808080")
        })
    
    return pd.DataFrame(rows).sort_values("%", ascending=False), final_total_area

def get_raster_img_b64(data, nodata):
    """Genera imagen Base64 para visualización rápida."""
    rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
    for val, hex_c in LAND_COVER_COLORS.items():
        if isinstance(hex_c, str):
            hex_c = hex_c.lstrip('#')
            r, g, b = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
            mask_val = (data == val)
            rgba[mask_val, 0] = r
            rgba[mask_val, 1] = g
            rgba[mask_val, 2] = b
            rgba[mask_val, 3] = 180 
    
    rgba[(data == 0) | (data == nodata), 3] = 0
    
    image_data = io.BytesIO()
    plt.imsave(image_data, rgba, format='png')
    image_data.seek(0)
    return f"data:image/png;base64,{base64.b64encode(image_data.read()).decode('utf-8')}"

# --- NUEVO: FUNCIONALIDADES PARA HOVER Y LEYENDA ---

def vectorize_raster(data, transform, crs, nodata):
    """
    Convierte el raster a Polígonos (GeoDataFrame) para permitir HOVER en el mapa.
    Nota: Se debe usar con rasters de baja resolución o recortados para no bloquear el navegador.
    """
    mask_arr = (data != nodata) & (data != 0)
    shapes_gen = shapes(data, mask=mask_arr, transform=transform)
    
    geoms = []
    values = []
    for geom, val in shapes_gen:
        geoms.append(shape(geom))
        values.append(val)
        
    if not geoms: return gpd.GeoDataFrame()
    
    gdf = gpd.GeoDataFrame({'ID': values}, geometry=geoms, crs=crs)
    gdf['Cobertura'] = gdf['ID'].map(lambda x: LAND_COVER_LEGEND.get(int(x), f"Clase {int(x)}"))
    gdf['Color'] = gdf['ID'].map(lambda x: LAND_COVER_COLORS.get(int(x), "#808080"))
    
    # Reproyectar a Lat/Lon para Folium
    return gdf.to_crs(epsg=4326)

def generate_legend_html():
    """Crea una leyenda HTML flotante para el mapa."""
    html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; 
        background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px; 
        font-size: 12px; max-height: 250px; overflow-y: auto;">
        <b>Leyenda Coberturas</b><br>
    """
    for id_cov, name in LAND_COVER_LEGEND.items():
        color = LAND_COVER_COLORS.get(id_cov, "#808080")
        html += f'<i style="background:{color}; width:12px; height:12px; float:left; margin-right:5px; opacity:0.8;"></i> {name}<br>'
    html += "</div>"
    return html

def get_tiff_bytes(data, transform, crs, nodata):
    """Genera bytes para descargar el archivo TIFF."""
    mem_file = io.BytesIO()
    with rasterio.open(
        mem_file, 'w', driver='GTiff',
        height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype,
        crs=crs, transform=transform, nodata=nodata
    ) as dst:
        dst.write(data, 1)
    mem_file.seek(0)
    return mem_file

# --- CÁLCULOS SCS ---
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