# modules/hydro_physics.py

import numpy as np
import rasterio
from scipy.interpolate import griddata
from pyproj import Transformer
import os

# --- A. BASE DE CONOCIMIENTO ---
CLC_C_BASE = {
    111: 0.90, 112: 0.85, 121: 0.85, # Urbano
    211: 0.60, 231: 0.50, 241: 0.45, # Cultivos
    311: 0.15, 321: 0.20, 312: 0.18, # Bosques
    322: 0.25, 511: 0.05, # Herbazales
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

# --- C. MUESTREO RASTER (EL FIX DE COORDENADAS) ---
def sample_raster_to_grid(raster_path, grid_x, grid_y):
    """
    Lee un raster y lo adapta a la malla, sin importar su proyección.
    """
    if not os.path.exists(raster_path):
        return np.zeros_like(grid_x)
        
    try:
        with rasterio.open(raster_path) as src:
            # 1. Preparar coordenadas de consulta (aplanadas)
            rows, cols = grid_x.shape
            lon_flat = grid_x.flatten()
            lat_flat = grid_y.flatten()
            
            # 2. DETECCIÓN Y TRANSFORMACIÓN DE CRS
            # El grid viene en WGS84 (EPSG:4326). Si el raster es diferente (ej. 3116), transformamos.
            if src.crs != 'EPSG:4326':
                try:
                    # Crear transformador: Desde Lat/Lon -> Hacia Raster
                    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    xx, yy = transformer.transform(lon_flat, lat_flat)
                    coords = list(zip(xx, yy))
                except:
                    # Fallback si falla pyproj
                    coords = list(zip(lon_flat, lat_flat))
            else:
                coords = list(zip(lon_flat, lat_flat))
            
            # 3. Muestrear
            sampled = np.array(list(src.sample(coords)))
            sampled = sampled.astype('float32')
            
            # Limpiar NoData (-9999 o valores negativos absurdos en DEM)
            if src.nodata is not None:
                sampled[sampled == src.nodata] = np.nan
            
            return sampled.flatten().reshape((rows, cols))
            
    except Exception as e:
        print(f"Error Raster: {e}")
        return np.zeros_like(grid_x)

# --- D. MODELO FÍSICO ---
def run_distributed_model(Z_P, grid_x, grid_y, paths):
    # 1. DEM (Ahora sí debería leerse gracias al Transformer)
    if paths.get('dem'):
        Z_Alt = sample_raster_to_grid(paths['dem'], grid_x, grid_y)
        # Limpieza agresiva de valores erróneos
        Z_Alt[Z_Alt < 0] = np.nan
        if np.all(np.isnan(Z_Alt)) or np.nanmax(Z_Alt) == 0:
            Z_Alt = np.full_like(Z_P, 1500) # Fallback si falla lectura
        else:
            # Rellenar huecos
            mask_ok = ~np.isnan(Z_Alt)
            if np.any(mask_ok):
                Z_Alt = griddata((grid_x[mask_ok], grid_y[mask_ok]), Z_Alt[mask_ok], (grid_x, grid_y), method='nearest')
    else:
        Z_Alt = np.full_like(Z_P, 1500)

    # Temperatura (Gradiente real)
    Z_T = np.maximum(28 - (0.006 * Z_Alt), 1.0)

    # 2. ETR (TURC)
    L = 300 + (25 * Z_T) + (0.05 * (Z_T**3))
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.sqrt(0.9 + (Z_P / L)**2)
        Z_ETR = np.minimum(Z_P / denom, Z_P)
        Z_ETR = np.nan_to_num(Z_ETR)
    
    Z_Exc = np.maximum(Z_P - Z_ETR, 0)

    # 3. COBERTURA Y ESCORRENTÍA
    Z_C = np.full_like(Z_P, 0.5)
    if paths.get('cobertura'):
        Z_Cob = sample_raster_to_grid(paths['cobertura'], grid_x, grid_y)
        if np.nanmax(Z_Cob) > 0:
            vfunc = np.vectorize(lambda x: CLC_C_BASE.get(int(x), 0.5))
            Z_C = vfunc(Z_Cob)

    # Pendiente (Slope)
    dy, dx = np.gradient(Z_Alt)
    Z_Slope = np.sqrt(dy**2 + dx**2)
    # Pendiente aumenta escorrentía
    Z_C_Mod = np.minimum(Z_C + (Z_Slope * 0.1), 0.95)
    
    Z_Q_Sup = Z_Exc * Z_C_Mod

    # 4. INFILTRACIÓN
    Z_Inf = np.maximum(Z_Exc - Z_Q_Sup, 0)
    
    # Recarga (Simplificada por ahora)
    Z_Recarga = Z_Inf * 0.3

    # 5. EROSIÓN (USLE Aprox)
    Z_C_Inv = np.maximum(1.0 - Z_C_Mod, 0.01) # Suelo desnudo erosiona más
    Z_Erosion = (Z_P * 0.5) * 0.3 * (1 + Z_Slope*5) * Z_C_Inv

    return {
        'P': Z_P, 'DEM': Z_Alt, 'T': Z_T,
        'ETR': Z_ETR, 'Q': Z_Q_Sup, 
        'Infiltracion': Z_Inf, 'Recarga': Z_Recarga,
        'Erosion': Z_Erosion, 'C_Escorrentia': Z_C_Mod
    }