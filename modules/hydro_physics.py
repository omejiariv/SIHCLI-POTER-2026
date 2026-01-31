# modules/hydro_physics.py

import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import griddata, Rbf
from rasterio.features import rasterize
import geopandas as gpd
import os

# --- A. BASE DE CONOCIMIENTO (DICCIONARIOS) ---

# Palabras clave para interpretar la columna 'caracteri' o 'litologia' de SUELOS
SOIL_KEYWORDS = {
    'arena': 0.50, 'arenoso': 0.50, 'fluvial': 0.45, 'aluvial': 0.45,
    'franco': 0.30, 'limo': 0.30,
    'arcilla': 0.10, 'arcilloso': 0.10, 'ceniza': 0.35,
    'roca': 0.02, 'duro': 0.05, 'ígnea': 0.05, 'metamórfica': 0.05,
    'conglomerado': 0.20
}

# Códigos Corine Land Cover (CLC) -> Coeficiente Escorrentía Base (C)
CLC_C_BASE = {
    111: 0.90, 112: 0.85, 121: 0.85, # Urbano
    211: 0.60, 231: 0.50, 241: 0.45, # Cultivos
    311: 0.15, 321: 0.20, 312: 0.18, # Bosques
    322: 0.25, 511: 0.05, # Herbazales / Agua
    'default': 0.50
}

# --- B. MOTOR DE INTERPOLACIÓN (LA FUNCIÓN QUE FALTABA) ---

def interpolar_variable(gdf_puntos, columna_valor, grid_x, grid_y, metodo='linear'):
    """
    Genera una superficie continua a partir de puntos (Estaciones).
    Esta es la función que el sistema no encontraba.
    """
    points_x = gdf_puntos.geometry.x.values
    points_y = gdf_puntos.geometry.y.values
    values = gdf_puntos[columna_valor].values
    
    try:
        # Intento 1: Lineal (Rápido y seguro)
        Z = griddata((points_x, points_y), values, (grid_x, grid_y), method=metodo)
        
        # Rellenar huecos (NaNs en bordes convexos) con Nearest
        if np.any(np.isnan(Z)):
            Z_near = griddata((points_x, points_y), values, (grid_x, grid_y), method='nearest')
            Z[np.isnan(Z)] = Z_near[np.isnan(Z)]
            
        return np.maximum(Z, 0) # Física: No valores negativos
    except Exception as e:
        print(f"Error interpolación: {e}")
        return np.zeros_like(grid_x)

# --- C. PROCESAMIENTO ESPACIAL (RASTER SAMPLING) ---

def sample_raster_to_grid(raster_path, grid_x, grid_y):
    """Lee un GeoTIFF y lo remuestrea a la malla de análisis."""
    if not os.path.exists(raster_path):
        return np.zeros_like(grid_x)
        
    try:
        with rasterio.open(raster_path) as src:
            # Aplanar coordenadas para muestreo masivo
            coords = list(zip(grid_x.flatten(), grid_y.flatten()))
            # Muestrear (devuelve generador)
            sampled = np.array(list(src.sample(coords)))
            # Reconstruir forma original (grid_x.shape)
            # Sampled suele venir como (N, 1) -> aplanar a (N,) -> reshape
            return sampled.flatten().reshape(grid_x.shape)
    except Exception as e:
        print(f"Error leyendo raster {raster_path}: {e}")
        return np.zeros_like(grid_x)

# --- D. MOTOR FÍSICO (BALANCE DISTRIBUIDO) ---

def run_distributed_model(Z_P, grid_x, grid_y, paths, vector_data=None):
    """
    Ejecuta el modelo físico completo (Turc + Balance + USLE).
    """
    # 1. ELEVACIÓN (DEM) & TEMPERATURA
    if paths.get('dem') and os.path.exists(paths['dem']):
        Z_Alt = sample_raster_to_grid(paths['dem'], grid_x, grid_y)
        Z_Alt[Z_Alt < -100] = np.nan # Limpiar NoData
        # Rellenar huecos
        if np.any(np.isnan(Z_Alt)):
            mask_ok = ~np.isnan(Z_Alt)
            if np.any(mask_ok):
                Z_Alt = griddata((grid_x[mask_ok], grid_y[mask_ok]), Z_Alt[mask_ok], (grid_x, grid_y), method='nearest')
            else:
                Z_Alt = np.full_like(Z_P, 1500)
    else:
        Z_Alt = np.full_like(Z_P, 1500) # Fallback altitud media

    # Gradiente Térmico
    Z_T = np.maximum(28 - (0.006 * Z_Alt), 1.0)

    # 2. ETR (TURC)
    L = 300 + (25 * Z_T) + (0.05 * (Z_T**3))
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.sqrt(0.9 + (Z_P / L)**2)
        Z_ETR = np.minimum(Z_P / denom, Z_P)
        Z_ETR = np.nan_to_num(Z_ETR)
    
    Z_Exc = np.maximum(Z_P - Z_ETR, 0)

    # 3. ESCORRENTÍA (COBERTURA + PENDIENTE)
    Z_C = np.full_like(Z_P, 0.5)
    if paths.get('cobertura') and os.path.exists(paths['cobertura']):
        Z_Cob_Code = sample_raster_to_grid(paths['cobertura'], grid_x, grid_y)
        vfunc = np.vectorize(lambda x: CLC_C_BASE.get(int(x), 0.5))
        Z_C = vfunc(Z_Cob_Code)

    # Pendiente (Slope)
    dy, dx = np.gradient(Z_Alt)
    Z_Slope = np.sqrt(dy**2 + dx**2) 
    Z_C_Mod = np.minimum(Z_C + (Z_Slope * 0.05), 0.95)

    # Q Superficial
    Z_Q_Sup = Z_Exc * Z_C_Mod

    # 4. INFILTRACIÓN Y RECARGA
    Z_Inf = np.maximum(Z_Exc - Z_Q_Sup, 0)
    
    # Factor Kp (Permeabilidad del Suelo) - Simplificado por ahora
    Z_Kp = np.full_like(Z_P, 0.3)
    Z_Recarga = Z_Inf * Z_Kp

    # 5. EROSIÓN (USLE)
    Z_R_Factor = Z_P * 0.5 
    Z_LS_Factor = 1 + (Z_Slope * 2)
    Z_K_Factor = 0.3 
    Z_C_Factor = np.maximum(0.5 - (Z_C * 0.4), 0.01)
    
    Z_Erosion = Z_R_Factor * Z_K_Factor * Z_LS_Factor * Z_C_Factor

    return {
        'P': Z_P, 'DEM': Z_Alt, 'T': Z_T,
        'ETR': Z_ETR, 'Q': Z_Q_Sup,
        'Infiltracion': Z_Inf, 'Recarga': Z_Recarga,
        'Erosion': Z_Erosion,
        'C_Escorrentia': Z_C_Mod
    }