# modules/hydro_physics.py

import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import griddata
from rasterio.features import rasterize
import geopandas as gpd

# --- A. BASE DE CONOCIMIENTO (DICCIONARIOS) ---

# Palabras clave para interpretar la columna 'caracteri' o 'litologia' de SUELOS
# Asigna factor Kp (percolación profunda): 0.0 (Impermeable) a 1.0 (Arena pura)
SOIL_KEYWORDS = {
    'arena': 0.50, 'arenoso': 0.50, 'fluvial': 0.45, 'aluvial': 0.45,
    'franco': 0.30, 'limo': 0.30,
    'arcilla': 0.10, 'arcilloso': 0.10, 'ceniza': 0.35,
    'roca': 0.02, 'duro': 0.05, 'ígnea': 0.05, 'metamórfica': 0.05,
    'conglomerado': 0.20
}

# Códigos Corine Land Cover (CLC) -> Coeficiente Escorrentía Base (C)
# C alto = Escurre mucho (Ciudad). C bajo = Infiltra mucho (Bosque)
CLC_C_BASE = {
    111: 0.90, 112: 0.85, 121: 0.85, # Urbano
    211: 0.60, 231: 0.50, 241: 0.45, # Cultivos
    311: 0.15, 321: 0.20, 312: 0.18, # Bosques
    322: 0.25, 511: 0.05, # Herbazales / Agua
    'default': 0.50
}

# --- B. PROCESAMIENTO ESPACIAL (RASTER SAMPLING) ---

def sample_raster_to_grid(raster_path, grid_x, grid_y):
    """Lee un GeoTIFF y lo remuestrea a la malla de análisis (grid_x, grid_y)."""
    try:
        with rasterio.open(raster_path) as src:
            # Aplanar coordenadas para muestreo masivo
            coords = list(zip(grid_x.flatten(), grid_y.flatten()))
            # Muestrear (devuelve generador)
            sampled = np.array(list(src.sample(coords)))
            # Reconstruir forma original
            return sampled.reshape(grid_x.shape)
    except Exception as e:
        print(f"Error leyendo raster {raster_path}: {e}")
        return np.zeros_like(grid_x)

def rasterize_vector_to_grid(gdf_vector, column, grid_x, grid_y, transform, shape):
    """Convierte GeoJSON a Matriz usando una columna descriptiva."""
    try:
        # Función auxiliar para mapear texto a valor numérico
        def get_val(text):
            if not isinstance(text, str): return 0.3
            text = text.lower()
            for key, val in SOIL_KEYWORDS.items():
                if key in text: return val
            return 0.3 # Default Franco

        # Crear lista de tuplas (geometria, valor)
        shapes = ((geom, get_val(row[column])) for geom, row in zip(gdf_vector.geometry, gdf_vector[column]))
        
        # Rasterizar
        arr = rasterize(
            shapes=shapes,
            out_shape=shape,
            transform=transform,
            fill=0.3, # Valor por defecto si no hay polígono
            dtype='float32'
        )
        # Nota: rasterize devuelve array orientado a imagen. Puede requerir flip si grid_x/y son cartesianos.
        # Asumiremos alineación por transform.
        return arr.T # Transpuesta suele ser necesaria entre Rasterio(rc) y Meshgrid(xy)
    except Exception as e:
        print(f"Error rasterizando vector: {e}")
        return np.full_like(grid_x, 0.3)

# --- C. MOTOR FÍSICO (BALANCE DISTRIBUIDO) ---

def run_distributed_model(Z_P, grid_x, grid_y, paths, vector_data=None):
    """
    Ejecuta el modelo físico completo.
    Args:
        Z_P: Matriz de Precipitación interpolada.
        grid_x, grid_y: Mallas de coordenadas.
        paths: Dict con rutas {'dem': '...', 'cobertura': '...'}.
        vector_data: Dict con GeoDataFrames {'suelos': gdf}.
    """
    # 1. ELEVACIÓN (DEM) & TEMPERATURA
    if paths.get('dem'):
        Z_Alt = sample_raster_to_grid(paths['dem'], grid_x, grid_y)
        # Fix: Si el DEM tiene nodata (-9999), limpiar
        Z_Alt[Z_Alt < -100] = np.nan
        # Rellenar huecos con interpolación cercana
        if np.any(np.isnan(Z_Alt)):
            mask_ok = ~np.isnan(Z_Alt)
            Z_Alt = griddata((grid_x[mask_ok], grid_y[mask_ok]), Z_Alt[mask_ok], (grid_x, grid_y), method='nearest')
    else:
        Z_Alt = np.full_like(Z_P, 1500) # Fallback

    # Gradiente Térmico: T disminuye 0.6°C por cada 100m
    Z_T = np.maximum(28 - (0.006 * Z_Alt), 1.0)

    # 2. ETR (TURC)
    L = 300 + (25 * Z_T) + (0.05 * (Z_T**3))
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.sqrt(0.9 + (Z_P / L)**2)
        Z_ETR = np.minimum(Z_P / denom, Z_P) # ETR <= P
        Z_ETR = np.nan_to_num(Z_ETR)
    
    # Excedente Bruto (P - ETR)
    Z_Exc = np.maximum(Z_P - Z_ETR, 0)

    # 3. ESCORRENTÍA (COBERTURA + PENDIENTE)
    # Cobertura Base
    Z_C = np.full_like(Z_P, 0.5)
    if paths.get('cobertura'):
        Z_Cob_Code = sample_raster_to_grid(paths['cobertura'], grid_x, grid_y)
        # Mapeo vectorizado
        vfunc = np.vectorize(lambda x: CLC_C_BASE.get(int(x), 0.5))
        Z_C = vfunc(Z_Cob_Code)

    # Factor Pendiente (Slope)
    dy, dx = np.gradient(Z_Alt) # Unidades: m/pixel
    # Necesitamos saber el tamaño del pixel en metros para pendiente real
    # Aprox: 1 grado ~ 111000m. 
    # Simplificación: Pendiente relativa adimensional
    Z_Slope = np.sqrt(dy**2 + dx**2) 
    # Ajuste: A mayor pendiente, mayor escorrentía (C aumenta)
    # Factor de corrección empírico: +0.01 por cada unidad de gradiente
    Z_C_Mod = np.minimum(Z_C + (Z_Slope * 0.05), 0.95)

    # Cálculo Q Superficial
    Z_Q_Sup = Z_Exc * Z_C_Mod

    # 4. INFILTRACIÓN Y RECARGA (SUELOS)
    Z_Inf = np.maximum(Z_Exc - Z_Q_Sup, 0)
    
    # Factor Kp (Permeabilidad del Suelo)
    Z_Kp = np.full_like(Z_P, 0.3)
    # Si tuviéramos rasterización de suelos, iría aquí.
    # Por ahora, usamos el DEM como proxy: Zonas bajas (depósitos) más permeables que zonas altas (roca)
    # Esto es una heurística hasta rasterizar el GeoJSON de suelos completamente
    # Z_Kp = 0.5 - (Z_Alt / 8000) # Ejemplo dummy
    
    # Si tenemos el vector de suelos cargado, lo usamos (Rasterización pendiente de integración completa)
    # Z_Recarga = Z_Inf * Z_Kp
    Z_Recarga = Z_Inf * 0.5 # Asumiendo suelo franco medio temporalmente

    # 5. EROSIÓN (USLE SIMPLIFICADO)
    # R (Lluvia), K (Suelo), LS (Pendiente), C (Cobertura)
    # R derivado de P (Fournier approx: R ~ P^2 / P_anual -> P)
    Z_R_Factor = Z_P * 0.5 
    Z_LS_Factor = 1 + (Z_Slope * 2)
    Z_K_Factor = 0.3 # Suelo medio
    Z_C_Factor = np.maximum(0.5 - (Z_C * 0.4), 0.01) # Inverso de escorrentía: Bosque C bajo escorrentía, bajo C erosión
    
    Z_Erosion = Z_R_Factor * Z_K_Factor * Z_LS_Factor * Z_C_Factor

    return {
        'P': Z_P, 'DEM': Z_Alt, 'T': Z_T,
        'ETR': Z_ETR, 'Q': Z_Q_Sup,
        'Infiltracion': Z_Inf, 'Recarga': Z_Recarga,
        'Erosion': Z_Erosion,
        'C_Escorrentia': Z_C_Mod
    }