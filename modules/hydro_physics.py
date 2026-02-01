# modules/hydro_physics.py

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import os
from modules.interpolation import interpolador_maestro

# --- A. BASE DE CONOCIMIENTO (Sin Cambios) ---
CLC_C_BASE = {
    111: 0.90, 112: 0.85, 121: 0.85, # Urbanos
    211: 0.60, 231: 0.50, 241: 0.45, # Agrícolas
    311: 0.15, 321: 0.20, 312: 0.18, # Bosques
    322: 0.25, 511: 0.05,            # Herbazales/Páramo
    'default': 0.50
}

# --- B. INTERPOLACIÓN AVANZADA (IDW) ---
def idw_interpolation(x, y, z, grid_x, grid_y, power=2):
    """
    Interpolación Inverse Distance Weighting (IDW).
    Elimina el efecto de 'líneas rectas' y 'triángulos'.
    """
    # Aplanamos las coordenadas de la grilla
    xi = grid_x.flatten()
    yi = grid_y.flatten()
    
    # Coordenadas de las estaciones
    xi_st = x
    yi_st = y
    zi_st = z
    
    # Calculamos distancias (Broadcasting)
    # Distancia entre cada pixel y cada estación
    dist = np.sqrt((xi[:, None] - xi_st[None, :])**2 + (yi[:, None] - yi_st[None, :])**2)
    
    # Evitar división por cero (si un pixel cae exacto en una estación)
    dist = np.where(dist == 0, 1e-10, dist)
    
    # Pesos
    weights = 1.0 / dist**power
    
    # Suma ponderada
    z_interp = np.sum(weights * zi_st, axis=1) / np.sum(weights, axis=1)
    
    return z_interp.reshape(grid_x.shape)

def interpolar_variable(gdf_puntos, columna_valor, grid_x, grid_y, method='kriging', dem_array=None):
    """
    Función puente que llama al interpolador maestro de modules/interpolation.py
    """
    # Llamamos a la función que acabamos de crear/potenciar
    Z_Interp, Z_Error = interpolador_maestro(
        df_puntos=gdf_puntos,
        col_val=columna_valor,
        grid_x=grid_x,
        grid_y=grid_y,
        metodo=method,
        dem_grid=dem_array # Pasamos el DEM por si se usa KED
    )
    
    # Saneamiento final para física
    Z_Interp = np.nan_to_num(Z_Interp, nan=0)
    return np.maximum(Z_Interp, 0), Z_Error


# --- C. WARPING (Sin Cambios) ---
def warper_raster_to_grid(raster_path, bounds, shape):
    if not raster_path or not os.path.exists(raster_path):
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

# --- D. MOTOR FÍSICO ---
def run_distributed_model(Z_P, grid_x, grid_y, paths, bounds):
    shape = grid_x.shape
    
    # 1. DEM
    Z_Alt = np.full_like(Z_P, 1500) # Default
    if paths.get('dem'):
        Z_dem_raw = warper_raster_to_grid(paths['dem'], bounds, shape)
        # Si el raster tiene datos válidos, los usamos
        if np.nanmax(Z_dem_raw) > 0:
            Z_Alt = np.nan_to_num(Z_dem_raw, nan=np.nanmean(Z_dem_raw))

    # Temperatura (Gradiente altitudinal)
    Z_T = np.maximum(28 - (0.006 * Z_Alt), 1.0)

    # 2. ETR (TURC)
    L = 300 + (25 * Z_T) + (0.05 * (Z_T**3))
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.sqrt(0.9 + (Z_P / L)**2)
        Z_ETR = np.nan_to_num(np.minimum(Z_P / denom, Z_P))
    
    Z_Exc = np.maximum(Z_P - Z_ETR, 0)

    # 3. COBERTURA
    Z_C = np.full_like(Z_P, 0.5)
    if paths.get('cobertura'):
        Z_Cob = warper_raster_to_grid(paths['cobertura'], bounds, shape)
        Z_Cob_Safe = np.nan_to_num(Z_Cob, nan=0)
        
        if np.nanmax(Z_Cob_Safe) > 0:
            # Vectorización optimizada
            keys = np.array(list(CLC_C_BASE.keys())[:-1]) # Quitamos 'default'
            vals = np.array(list(CLC_C_BASE.values())[:-1])
            
            # Mapeo rápido usando numpy searchsorted o similar sería ideal, 
            # pero el método map/vectorize es seguro por ahora.
            def map_c(code): return CLC_C_BASE.get(int(code), 0.5)
            vfunc = np.vectorize(map_c)
            Z_C = vfunc(Z_Cob_Safe)

    # 4. PENDIENTE
    scale_factor = 111000 
    dy, dx = np.gradient(Z_Alt)
    cell_size_y = np.abs(grid_y[1,0] - grid_y[0,0]) * scale_factor
    cell_size_x = np.abs(grid_x[0,1] - grid_x[0,0]) * scale_factor
    
    denom_slope_x = np.where(cell_size_x == 0, 1, cell_size_x)
    denom_slope_y = np.where(cell_size_y == 0, 1, cell_size_y)

    Z_Slope_Pct = np.sqrt((dy/denom_slope_y)**2 + (dx/denom_slope_x)**2)
    
    # Ajuste de Escorrentía por Pendiente
    Z_C_Mod = np.minimum(Z_C + (Z_Slope_Pct * 0.2), 0.95) # Factor 0.2 para no saturar

    Z_Q_Sup = Z_Exc * Z_C_Mod

    # 5. INFILTRACIÓN
    Z_Inf = np.maximum(Z_Exc - Z_Q_Sup, 0)
    Z_Rec_Pot = Z_Inf 
    Z_Rec_Real = Z_Inf * 0.3 # Factor geológico simple

    # 6. RENDIMIENTO
    Z_Q_Total_mm = Z_Q_Sup + Z_Rec_Real
    Z_Rendimiento = Z_Q_Total_mm * 10 # m3/ha

    # 7. EROSIÓN
    Z_C_Inv = np.maximum(1.0 - Z_C_Mod, 0.01)
    Z_Erosion = (Z_P * 0.5) * 0.3 * (1 + Z_Slope_Pct * 5) * Z_C_Inv

    return {
        'P': Z_P, 'DEM': Z_Alt, 'T': Z_T, 'ETR': Z_ETR, 
        'Q': Z_Q_Sup, 'Infiltracion': Z_Inf, 
        'Recarga_Pot': Z_Rec_Pot, 'Recarga_Real': Z_Rec_Real,
        'Rendimiento': Z_Rendimiento, 'Erosion': Z_Erosion, 'C_Escorrentia': Z_C_Mod
    }