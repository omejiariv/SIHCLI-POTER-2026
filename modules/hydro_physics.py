# modules/hydro_physics.py

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
import os

# --- A. BASE DE CONOCIMIENTO ---
CLC_C_BASE = {
    # Urbanos (Alta escorrentía)
    111: 0.90, 112: 0.85, 121: 0.85, 
    # Agrícolas
    211: 0.60, 231: 0.50, 241: 0.45, 
    # Bosques (Baja escorrentía)
    311: 0.15, 321: 0.20, 312: 0.18, 
    # Herbazales / Páramo
    322: 0.25, 511: 0.05,
    'default': 0.50
}

# --- B. INTERPOLACIÓN ---
def interpolar_variable(gdf_puntos, columna_valor, grid_x, grid_y):
    points_x = gdf_puntos.geometry.x.values
    points_y = gdf_puntos.geometry.y.values
    values = gdf_puntos[columna_valor].values
    try:
        Z = griddata((points_x, points_y), values, (grid_x, grid_y), method='linear')
        if np.any(np.isnan(Z)):
            Z_near = griddata((points_x, points_y), values, (grid_x, grid_y), method='nearest')
            Z[np.isnan(Z)] = Z_near[np.isnan(Z)]
        return np.maximum(Z, 0)
    except: return np.zeros_like(grid_x)

# --- C. WARPING (SIN CAMBIOS) ---
def warper_raster_to_grid(raster_path, bounds, shape):
    if not os.path.exists(raster_path):
        return np.zeros(shape)
    
    minx, miny, maxx, maxy = bounds
    height, width = shape

    try:
        with rasterio.open(raster_path) as src:
            dst_transform = from_bounds(minx, miny, maxx, maxy, width, height)
            destination = np.zeros(shape, dtype=np.float32)
            
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs='EPSG:4326',
                resampling=Resampling.bilinear,
                src_nodata=src.nodata if src.nodata else -9999,
                dst_nodata=np.nan 
            )
            return destination
    except: return np.zeros(shape)

# --- D. MOTOR FÍSICO (FIXED & UPGRADED) ---
def run_distributed_model(Z_P, grid_x, grid_y, paths, bounds):
    shape = grid_x.shape
    
    # 1. DEM
    if paths.get('dem'):
        Z_Alt = warper_raster_to_grid(paths['dem'], bounds, shape)
        if np.nanmax(Z_Alt) == 0: Z_Alt = np.full_like(Z_P, 1500)
        else:
            # Rellenar huecos DEM con promedio
            Z_Alt = np.nan_to_num(Z_Alt, nan=np.nanmean(Z_Alt))
    else:
        Z_Alt = np.full_like(Z_P, 1500)

    # Temperatura
    Z_T = np.maximum(28 - (0.006 * Z_Alt), 1.0)

    # 2. ETR (TURC)
    L = 300 + (25 * Z_T) + (0.05 * (Z_T**3))
    with np.errstate(divide='ignore'):
        denom = np.sqrt(0.9 + (Z_P / L)**2)
        Z_ETR = np.nan_to_num(np.minimum(Z_P / denom, Z_P))
    
    Z_Exc = np.maximum(Z_P - Z_ETR, 0)

    # 3. COBERTURA (FIX DEL ERROR NaN)
    Z_C = np.full_like(Z_P, 0.5)
    
    if paths.get('cobertura'):
        Z_Cob = warper_raster_to_grid(paths['cobertura'], bounds, shape)
        
        # --- FIX CRÍTICO ---
        # Reemplazamos NaN por 0 antes de convertir a entero
        Z_Cob_Safe = np.nan_to_num(Z_Cob, nan=0)
        
        if np.nanmax(Z_Cob_Safe) > 0:
            # Función segura que maneja el mapeo
            def map_c(code):
                return CLC_C_BASE.get(int(code), 0.5)
            
            vfunc = np.vectorize(map_c)
            Z_C = vfunc(Z_Cob_Safe)

    # 4. PENDIENTE Y ESCORRENTÍA
    scale_factor = 111000 
    dy, dx = np.gradient(Z_Alt)
    cell_size_y = np.abs(grid_y[1,0] - grid_y[0,0]) * scale_factor
    cell_size_x = np.abs(grid_x[0,1] - grid_x[0,0]) * scale_factor
    
    # Evitar división por cero
    cell_size_x = cell_size_x if cell_size_x > 0 else 1
    cell_size_y = cell_size_y if cell_size_y > 0 else 1

    Z_Slope_Pct = np.sqrt((dy/cell_size_y)**2 + (dx/cell_size_x)**2)
    Z_C_Mod = np.minimum(Z_C + (Z_Slope_Pct * 0.5), 0.95)

    Z_Q_Sup = Z_Exc * Z_C_Mod

    # 5. INFILTRACIÓN Y RECARGAS
    Z_Inf = np.maximum(Z_Exc - Z_Q_Sup, 0)
    
    # Recarga Potencial: Todo lo que se infiltra (Teórico)
    Z_Rec_Pot = Z_Inf 
    
    # Recarga Real: Infiltración limitada por factor geológico (simplificado 30% por ahora)
    # Idealmente, esto vendría de un mapa de hidrogeología
    Z_Rec_Real = Z_Inf * 0.3 

    # 6. RENDIMIENTO HÍDRICO (m3/ha-año)
    # Q total (mm) = Q_Sup + Flujo Base (aprox igual a Recarga Real)
    Z_Q_Total_mm = Z_Q_Sup + Z_Rec_Real
    # Conversión: 1 mm = 10 m3/ha
    Z_Rendimiento = Z_Q_Total_mm * 10

    # 7. EROSIÓN (Riesgo)
    Z_C_Inv = np.maximum(1.0 - Z_C_Mod, 0.01)
    Z_Erosion = (Z_P * 0.5) * 0.3 * (1 + Z_Slope_Pct * 10) * Z_C_Inv

    return {
        'P': Z_P, 
        'DEM': Z_Alt, 
        'T': Z_T,
        'ETR': Z_ETR, 
        'Q': Z_Q_Sup, 
        'Infiltracion': Z_Inf, 
        'Recarga_Pot': Z_Rec_Pot,
        'Recarga_Real': Z_Rec_Real,
        'Rendimiento': Z_Rendimiento,
        'Erosion': Z_Erosion, 
        'C_Escorrentia': Z_C_Mod
    }