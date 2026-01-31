# modules/hydro_physics.py

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
import os

# --- A. BASE DE CONOCIMIENTO ---
CLC_C_BASE = {
    # Urbanos
    111: 0.90, 112: 0.85, 121: 0.85, 
    # Agrícolas
    211: 0.60, 231: 0.50, 241: 0.45, 
    # Bosques
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

# --- C. REPROYECCIÓN DE RASTER (LA SOLUCIÓN) ---
def warper_raster_to_grid(raster_path, bounds, shape):
    """
    Reproyecta y recorta un raster físico para que encaje PERFECTAMENTE 
    en la grilla de análisis (WGS84).
    
    Args:
        raster_path: Ruta al archivo .tif
        bounds: Tupla (minx, miny, maxx, maxy) en WGS84
        shape: Tupla (alto, ancho) de la grilla destino
    """
    if not os.path.exists(raster_path):
        return np.zeros(shape)

    minx, miny, maxx, maxy = bounds
    height, width = shape

    try:
        with rasterio.open(raster_path) as src:
            # 1. Definir la transformación destino (WGS84)
            dst_transform = from_bounds(minx, miny, maxx, maxy, width, height)
            dst_crs = 'EPSG:4326' # Forzamos WGS84 para coincidir con Folium
            
            # 2. Crear array destino vacío
            destination = np.zeros(shape, dtype=np.float32)

            # 3. La Magia: Reproject (Warp)
            # Convierte de Magna (src.crs) a WGS84 (dst_crs) automáticamente
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                # Usar valor nodata del origen o 0
                src_nodata=src.nodata if src.nodata is not None else -9999,
                dst_nodata=np.nan 
            )
            
            return destination

    except Exception as e:
        print(f"Error Warping {raster_path}: {e}")
        return np.zeros(shape)

# --- D. MODELO FÍSICO ---
def run_distributed_model(Z_P, grid_x, grid_y, paths, bounds):
    """
    Ejecuta el modelo usando Warping para alinear capas.
    """
    shape = grid_x.shape # (alto, ancho)
    
    # 1. DEM (ELEVACIÓN)
    if paths.get('dem'):
        Z_Alt = warper_raster_to_grid(paths['dem'], bounds, shape)
        # Limpieza: Si todo es 0 o NaN, el warping falló o no cubrió el área
        if np.nanmax(Z_Alt) == 0:
            Z_Alt = np.full_like(Z_P, 1500) # Fallback
        else:
            # Rellenar huecos visuales con promedio local
            mask_nan = np.isnan(Z_Alt) | (Z_Alt < 0)
            if np.any(mask_nan):
                Z_Alt[mask_nan] = np.nanmean(Z_Alt)
    else:
        Z_Alt = np.full_like(Z_P, 1500)

    # Temperatura (Gradiente)
    Z_T = np.maximum(28 - (0.006 * Z_Alt), 1.0)

    # 2. ETR (TURC)
    L = 300 + (25 * Z_T) + (0.05 * (Z_T**3))
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.sqrt(0.9 + (Z_P / L)**2)
        Z_ETR = np.minimum(Z_P / denom, Z_P)
        Z_ETR = np.nan_to_num(Z_ETR)
    
    Z_Exc = np.maximum(Z_P - Z_ETR, 0)

    # 3. COBERTURA (RASTER)
    Z_C = np.full_like(Z_P, 0.5) # Base default
    
    if paths.get('cobertura'):
        Z_Cob = warper_raster_to_grid(paths['cobertura'], bounds, shape)
        if np.nanmax(Z_Cob) > 0:
            # Vectorizar el mapeo de códigos
            vfunc = np.vectorize(lambda x: CLC_C_BASE.get(int(x), 0.5))
            Z_C = vfunc(Z_Cob)

    # 4. PENDIENTE (SLOPE)
    # Nota: grid_x, grid_y están en grados. 
    # Factor de corrección aproximado: 1 grado ~ 111,000 metros en el ecuador
    scale_factor = 111000 
    dy, dx = np.gradient(Z_Alt)
    # Pendiente adimensional (m/m)
    # dy (m) / tamaño_celda_y (grados) * scale
    # Aproximación robusta para visualización
    cell_size_y = np.abs(grid_y[1,0] - grid_y[0,0]) * scale_factor
    cell_size_x = np.abs(grid_x[0,1] - grid_x[0,0]) * scale_factor
    
    slope_y = dy / cell_size_y
    slope_x = dx / cell_size_x
    Z_Slope_Pct = np.sqrt(slope_y**2 + slope_x**2)
    
    # Modificar C con la pendiente (Si pendiente > 20%, C aumenta)
    Z_C_Mod = np.minimum(Z_C + (Z_Slope_Pct * 0.5), 0.95)

    # 5. BALANCE
    Z_Q_Sup = Z_Exc * Z_C_Mod
    Z_Inf = np.maximum(Z_Exc - Z_Q_Sup, 0)
    
    # Recarga (Simplificada)
    Z_Recarga = Z_Inf * 0.3

    # 6. EROSIÓN (USLE Aprox)
    # R(Lluvia) * K(Suelo const) * LS(Pendiente) * C(Cobertura inversa)
    Z_C_Inv = np.maximum(1.0 - Z_C_Mod, 0.01)
    Z_Erosion = (Z_P * 0.5) * 0.3 * (1 + Z_Slope_Pct * 10) * Z_C_Inv

    return {
        'P': Z_P, 'DEM': Z_Alt, 'T': Z_T,
        'ETR': Z_ETR, 'Q': Z_Q_Sup, 
        'Infiltracion': Z_Inf, 'Recarga': Z_Recarga,
        'Erosion': Z_Erosion, 'C_Escorrentia': Z_C_Mod
    }