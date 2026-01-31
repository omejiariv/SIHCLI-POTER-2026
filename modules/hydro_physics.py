# modules/hydro_physics.py

import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import griddata
from pyproj import Transformer # CLAVE PARA EL FIX
import os

# --- A. BASE DE CONOCIMIENTO ---

SOIL_KEYWORDS = {
    'arena': 0.50, 'arenoso': 0.50, 'fluvial': 0.45, 'aluvial': 0.45,
    'franco': 0.30, 'limo': 0.30,
    'arcilla': 0.10, 'arcilloso': 0.10, 'ceniza': 0.35,
    'roca': 0.02, 'duro': 0.05, 'ígnea': 0.05,
    'conglomerado': 0.20
}

CLC_C_BASE = {
    111: 0.90, 112: 0.85, 121: 0.85, # Urbano
    211: 0.60, 231: 0.50, 241: 0.45, # Cultivos
    311: 0.15, 321: 0.20, 312: 0.18, # Bosques
    322: 0.25, 511: 0.05, # Herbazales
    'default': 0.50
}

# --- B. INTERPOLACIÓN ---

def interpolar_variable(gdf_puntos, columna_valor, grid_x, grid_y, metodo='linear'):
    points_x = gdf_puntos.geometry.x.values
    points_y = gdf_puntos.geometry.y.values
    values = gdf_puntos[columna_valor].values
    
    try:
        Z = griddata((points_x, points_y), values, (grid_x, grid_y), method=metodo)
        if np.any(np.isnan(Z)):
            Z_near = griddata((points_x, points_y), values, (grid_x, grid_y), method='nearest')
            Z[np.isnan(Z)] = Z_near[np.isnan(Z)]
        return np.maximum(Z, 0)
    except Exception as e:
        print(f"Error interpolación: {e}")
        return np.zeros_like(grid_x)

# --- C. MUESTREO RASTER INTELIGENTE (EL FIX) ---

def sample_raster_to_grid(raster_path, grid_x, grid_y):
    """
    Lee un raster y muestrea valores en las coordenadas del grid.
    FIX: Detecta el CRS del raster y reproyecta los puntos si es necesario.
    """
    if not os.path.exists(raster_path):
        print(f"Raster no encontrado: {raster_path}")
        return np.zeros_like(grid_x)
        
    try:
        with rasterio.open(raster_path) as src:
            # 1. Preparar coordenadas origen (WGS84 / EPSG:4326)
            rows, cols = grid_x.shape
            x_flat = grid_x.flatten()
            y_flat = grid_y.flatten()
            
            # 2. Verificar si necesitamos transformación
            # Asumimos que el grid siempre viene en EPSG:4326 (Lat/Lon)
            if src.crs != 'EPSG:4326':
                try:
                    # Crear transformador: De LatLon -> RasterCRS
                    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    # Transformar coordenadas
                    xx, yy = transformer.transform(x_flat, y_flat)
                    coords = list(zip(xx, yy))
                except Exception as proj_error:
                    print(f"Error reproyección: {proj_error}. Usando coords originales.")
                    coords = list(zip(x_flat, y_flat))
            else:
                coords = list(zip(x_flat, y_flat))
            
            # 3. Muestrear
            sampled = np.array(list(src.sample(coords)))
            
            # 4. Manejo de NoData (-9999, etc)
            # Convertimos a float para permitir NaNs
            sampled = sampled.astype('float32')
            if src.nodata is not None:
                sampled[sampled == src.nodata] = np.nan
            
            # Aplanar resultado (sample devuelve lista de arrays)
            sampled = sampled.flatten()
            
            return sampled.reshape((rows, cols))
            
    except Exception as e:
        print(f"Error crítico leyendo raster {raster_path}: {e}")
        return np.zeros_like(grid_x)

# --- D. MOTOR FÍSICO ---

def run_distributed_model(Z_P, grid_x, grid_y, paths, vector_data=None):
    """Ejecuta el modelo físico distribuido."""
    
    # 1. DEM & TEMPERATURA
    if paths.get('dem'):
        Z_Alt = sample_raster_to_grid(paths['dem'], grid_x, grid_y)
        # Limpieza de valores absurdos (mar o errores)
        Z_Alt[Z_Alt < -50] = np.nan 
        # Rellenar NaNs del DEM con promedio local o fallback
        if np.any(np.isnan(Z_Alt)):
            mean_h = np.nanmean(Z_Alt)
            Z_Alt = np.nan_to_num(Z_Alt, nan=mean_h)
    else:
        Z_Alt = np.full_like(Z_P, 1500)

    # Física: T baja con la altura
    Z_T = np.maximum(28 - (0.006 * Z_Alt), 1.0)

    # 2. ETR (TURC)
    L = 300 + (25 * Z_T) + (0.05 * (Z_T**3))
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.sqrt(0.9 + (Z_P / L)**2)
        Z_ETR = np.minimum(Z_P / denom, Z_P)
        Z_ETR = np.nan_to_num(Z_ETR)
    
    Z_Exc = np.maximum(Z_P - Z_ETR, 0)

    # 3. ESCORRENTÍA (COBERTURA + PENDIENTE)
    Z_C = np.full_like(Z_P, 0.5) # Base
    
    if paths.get('cobertura'):
        Z_Cob_Code = sample_raster_to_grid(paths['cobertura'], grid_x, grid_y)
        # Si el raster de cobertura falló (todo ceros), usar default
        if np.nanmax(Z_Cob_Code) > 0:
            vfunc = np.vectorize(lambda x: CLC_C_BASE.get(int(x), 0.5))
            Z_C = vfunc(Z_Cob_Code)

    # Pendiente desde el DEM
    dy, dx = np.gradient(Z_Alt)
    Z_Slope = np.sqrt(dy**2 + dx**2)
    # Aumentar C en pendientes fuertes (Agua corre más)
    Z_C_Mod = np.minimum(Z_C + (Z_Slope * 0.1), 0.95)

    Z_Q_Sup = Z_Exc * Z_C_Mod

    # 4. INFILTRACIÓN
    Z_Inf = np.maximum(Z_Exc - Z_Q_Sup, 0)
    
    # 5. RECARGA
    # Por ahora factor Kp constante hasta rasterizar suelos
    Z_Kp = np.full_like(Z_P, 0.3) 
    Z_Recarga = Z_Inf * Z_Kp

    # 6. EROSIÓN (USLE)
    # R depende de P, K depende de Suelo, LS de pendiente, C de cobertura
    # Invertir C hidrológico para tener C de erosión (aprox)
    # Bosque: C_hidro bajo (0.15) -> C_erosion bajo (0.001)
    # Suelo desnudo: C_hidro alto (0.9) -> C_erosion alto (1.0)
    Z_C_Erosion = np.maximum((Z_C_Mod - 0.1) ** 2, 0.01) 
    
    Z_Erosion = (Z_P * 0.5) * 0.3 * (1 + Z_Slope*5) * Z_C_Erosion

    return {
        'P': Z_P, 'DEM': Z_Alt, 'T': Z_T,
        'ETR': Z_ETR, 'Q': Z_Q_Sup,
        'Infiltracion': Z_Inf, 'Recarga': Z_Recarga,
        'Erosion': Z_Erosion,
        'C_Escorrentia': Z_C_Mod
    }